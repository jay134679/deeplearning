-- Homework 2: train.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
-- Trainging and valdiation code.


require 'nn'
require 'optim'
require 'torch'
require 'xlua'


function train_one_epoch(opt, trainData, optimState, model, criterion)
   model:training()

   local confusion = optim.ConfusionMatrix(10)

   local targets = nil
   if opt.no_cuda then
      targets = torch.FloatTensor(opt.batchSize)
   else
      targets = torch.CudaTensor(opt.batchSize)
   end
   -- creates a random permutation of numbers 1 though the size of the training data,
   -- then splits them into batches.
   local indices = torch.randperm(trainData.data:size(1)):long():split(opt.batchSize)
   -- remove last element so that all the batches have equal size
   indices[#indices] = nil

   local parameters, gradParameters = model:getParameters()
   
   local tic = torch.tic()  -- starts timer
   
   for t,v in ipairs(indices) do
      xlua.progress(t, #indices)

      local inputs = trainData.data:index(1, v)
      targets:copy(trainData.labels:index(1, v))
      
      local feval = function(x)
	 if x ~= parameters then parameters:copy(x) end
	 gradParameters:zero()
	 
	 local outputs = model:forward(inputs)
	 local f = criterion:forward(outputs, targets)
	 local df_do = criterion:backward(outputs, targets)
	 model:backward(inputs, df_do)
	 
	 confusion:batchAdd(outputs, targets)
	 
	 return f, gradParameters
      end
      optim.sgd(feval, parameters, optimState)
   end
   
   confusion:updateValids()
   local time_secs = torch.toc(tic)
   DEBUG(('Train accuracy: %.2f'..' %%\t time: %.2f s'):format(
	    confusion.totalValid * 100, time_secs))
   
   local train_acc = confusion.totalValid * 100
   return train_acc
end

function evaluate_model(validateData, model)
   model:evaluate()

   local confusion = optim.ConfusionMatrix(10)
   
   print('==> evaluating')
   -- Don't know why Jake adds the labels to the confusion matrix in batches of
   -- 25, but sure, why not.
   local batch_size = 25
   for i = 1,validateData.data:size(1),batch_size do
      local outputs = model:forward(validateData.data:narrow(1, i, batch_size))
      confusion:batchAdd(outputs, validateData.labels:narrow(1, i, batch_size))
   end

   confusion:updateValids()
   DEBUG(('Valdiation accuracy: %.2f %%'):format(confusion.totalValid * 100))
   return confusion
end


function log_validation_stats(valLogger, model, epoch, train_acc, val_confusion,
			      optimState, experiment_dir)
   local val_acc = val_confusion.totalValid * 100
   valLogger:add{train_acc, val_acc}
   valLogger:style{'-','-'}
   valLogger:plot() -- TODO this should save an eps file, but it doesn't.
   
   local base64im
   do
      os.execute(('convert -density 200 %s/val.log.eps %s/val.png'):format(
		    experiment_dir, experiment_dir))
      os.execute(('openssl base64 -in %s/val.png -out %s/val.base64'):format(
		    experiment_dir, experiment_dir))
      local f = io.open(experiment_dir..'/val.base64')
      if f then
	 base64im = f:read'*all'  -- TODO huh?
      end
   end
   
   local file = assert(io.open(experiment_dir..'/report.html', 'w'))
   file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(experiment_dir, epoch, base64im))
   for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
	 file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
   end
   file:write'</table><pre>\n'
   file:write(tostring(val_confusion)..'\n')
   file:write(tostring(model)..'\n')
   file:write'</pre></body></html>'
   file:close()
end

-- NOTE: The model passed here should just be the layer of the custom model we
-- created.
-- Since it takes several minutes to save a model file, only save every
-- model_save_freq epochs.
function maybe_save_model(model, epoch, model_save_freq, experiment_dir, val_percent_acc_last, val_percent_acc)
   epsilon = 0.001
   -- save model ever model_save_freq epochs, but only in validation accuracy improves.  
   if (epoch % model_save_freq == 0) and (val_percent_acc - val_percent_acc_last > epsilon) then
      local filename = paths.concat(experiment_dir, 'model.net')
      DEBUG('==> saving model to '..filename)
      torch.save(filename, model)
   end
end


-- This modifies model and val_logger.
-- This returns the percentage of samples that were correctly
-- classified on the validation set and the average number of
-- milliseconds per sample required to train the model.
function train_validate_max_epochs(opt, provider, model,
				   custom_model_layer_index, experiment_dir)
   local valLogger = optim.Logger(paths.concat(experiment_dir, 'val.log'))
   valLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
   valLogger.showPlot = false

   print('==> setting criterion')
   local criterion = nn.CrossEntropyCriterion()
   if not opt.no_cuda then
      criterion = criterion:cuda()
   end

   print('==> configuring optimizer')   
   local optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = opt.learningRateDecay,
   }

   local val_percent_acc_last = 0
   for epoch = 1,opt.max_epoch do
      local epoch_debug_str = "==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']'
      DEBUG(epoch_debug_str)
      -- drop learning rate every "epoch_step" epochs
      if epoch % opt.epoch_step == 0 then
	 optimState.learningRate = optimState.learningRate/2
      end
      
      print(provider.trainData)
      local train_acc = train_one_epoch(opt, provider.trainData, optimState,
					model, criterion)
      val_confusion = evaluate_model(provider.valData, model) -- TO DO
      
      log_validation_stats(valLogger, model, epoch, train_acc, val_confusion,
			   optimState, experiment_dir)
      
      val_percent_acc = val_confusion.totalValid*100
      print ("LAST  "..val_percent_acc_last.."  NEW  ".. val_percent_acc)
      -- TODO shouldn't this save only when it's the best model seen so far, not just the best one since we last checked?
      maybe_save_model(model:get(custom_model_layer_index), epoch, opt.model_save_freq, experiment_dir, val_percent_acc_last, val_percent_acc)
      val_percent_acc_last = val_percent_acc
   end
end


---- Pseudo Label stuff ----


-- TODO extra data 
-- TODO unlabeled proportion
function pseudo_train_one_epoch(opt, trainData, extraData, unlabeledProportion,
				optimState, model, criterion)
   model:training()

   local confusion = optim.ConfusionMatrix(10)
   local parameters, gradParameters = model:getParameters()

   -- labeled data
   
   -- creates a random permutation of numbers 1 though the size of the training data,
   -- then splits them into batches.
   local labeled_batch_size = math.floor(opt.batchSize*(1-unlabeledProportion)) 
   local labeled_indices = torch.randperm(trainData.data:size(1)):long():split(labeled_batch_size)
      -- remove last element so that all the batches have equal size
   labeled_indices[#labeled_indices] = nil -- TODO huh?

   local targets = nil
   if opt.no_cuda then
      targets = torch.FloatTensor(labeled_batch_size)
   else
      targets = torch.CudaTensor(labeled_batch_size)
   end

   local tic = torch.tic()  -- starts timer
   
   for t,v in ipairs(labeled_indices) do
      xlua.progress(t, #labeld_indices)
      
      local inputs = trainData.data:index(1, v)
      targets:copy(trainData.labels:index(1, v))
      
      local feval = function(x)
	 if x ~= parameters then parameters:copy(x) end
	 gradParameters:zero() -- TODO what does this do?
	 
	 local outputs = model:forward(inputs)
	 local error = criterion:forward(outputs, targets)
	 local error_gradient = criterion:backward(outputs, targets)
	 
	 model:backward(inputs, error_gradient)
	 
	 confusion:batchAdd(outputs, targets)
	 return error, gradParameters
      end
      optim.sgd(feval, parameters, optimState)
   end

   -- unlabeled data
   -- If kAnneal == 0, then the gradient update will be zero, so skip it.
   if kAnneal > 0 then   
      local unlabeled_batch_size = math.floor(opt.batchSize*(unlabeledProportion))
      unlabeled_indices[#unlabeled_indices] = nil -- TODO huh?
      
      local unlabeled_indices = torch.randperm(extraData.data:size(1)):long():split(unlabeled_batch_size)
      
      local targets = nil
      if opt.no_cuda then
	 targets = torch.FloatTensor(unlabeled_batch_size)
      else
	 targets = torch.CudaTensor(unlabeled_batch_size)
      end
      
      for t,v in ipairs(unlabeled_indices) do
	 xlua.progress(t, #unlabwled_indices)
	 
	 local inputs = extraData.data:index(1, v)
	 
	 local feval = function(x)
	    if x ~= parameters then parameters:copy(x) end
	    gradParameters:zero() -- TODO huh? Doesn't look like this is used.
	    
	    local outputs = model:forward(inputs)
	    
	    -- compute the pseudo labels.
	    local _, predictedLabels = torch.max(outputs, 2)
	    predictedLabels = predictedLabels:float()
	    targets:copy(predictedLabels)
	    
	    local error = criterion:forward(outputs, targets) * kAnneal
	     -- TODO I hope this is right...
	    local error_gradient = criterion:backward(outputs, targets) * kAnneal
	    model:backward(inputs, error_gradient)
	    
	    confusion:batchAdd(outputs, targets)
	    
	    return error, gradParameters
	 end
	 optim.sgd(feval, parameters, optimState)
      end
   end
      
   confusion:updateValids()
   local time_secs = torch.toc(tic)
   DEBUG(('Train accuracy: %.2f'..' %%\t time: %.2f s'):format(
	    confusion.totalValid * 100, time_secs))
   
   local train_acc = confusion.totalValid * 100
   return train_acc
end


-- This modifies model and val_logger.
-- This returns the percentage of samples that were correctly
-- classified on the validation set and the average number of
-- milliseconds per sample required to train the model.
function pseudo_train_validate_max_epochs(opt, provider, model,
				   custom_model_layer_index, experiment_dir)
   local valLogger = optim.Logger(paths.concat(experiment_dir, 'val.log'))
   valLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
   valLogger.showPlot = false

   print('==> setting criterion')
   local criterion = nn.CrossEntropyCriterion()
   if not opt.no_cuda then
      criterion = criterion:cuda()
   end

   print('==> configuring optimizer')   
   local optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = opt.learningRateDecay,
   }

   -- New Pseudo label stuff
   kAnneal = 0
   unlabeledProportion = provider.extraData.size() / (provider.extraData.size() + provider.trainData.size())
   
   local val_percent_acc_last = 0
   for epoch = 1,opt.max_epoch do
      local epoch_debug_str = "==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']'
      DEBUG(epoch_debug_str)
      -- drop learning rate every "epoch_step" epochs
      if epoch % opt.epoch_step == 0 then
	 optimState.learningRate = optimState.learningRate/2
      end
      
      print(provider.trainData)
      local train_acc = train_one_epoch(opt, provider.trainData, optimState,
					model, criterion)
      val_confusion = evaluate_model(provider.valData, model) -- TO DO
      
      log_validation_stats(valLogger, model, epoch, train_acc, val_confusion,
			   optimState, experiment_dir)
      
      val_percent_acc = val_confusion.totalValid*100
      print ("LAST  "..val_percent_acc_last.."  NEW  ".. val_percent_acc)
      -- TODO shouldn't this save only when it's the best model seen so far, not just the best one since we last checked?
      maybe_save_model(model:get(custom_model_layer_index), epoch, opt.model_save_freq, experiment_dir, val_percent_acc_last, val_percent_acc)
      val_percent_acc_last = val_percent_acc
   end
end

