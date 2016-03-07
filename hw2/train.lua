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
   -- TODO I don't know why it's important for all batches to have the same size...
   -- remove last element so that all the batches have equal size
   if #indices > 1 then
      indices[#indices] = nil
   end

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
      return val_percent_acc

   else
       return val_percent_acc_last 
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
      
      local train_acc = train_one_epoch(opt, provider.trainData, optimState,
					model, criterion)
      val_confusion = evaluate_model(provider.valData, model) -- TO DO
      
      log_validation_stats(valLogger, model, epoch, train_acc, val_confusion,
			   optimState, experiment_dir)
      
      val_percent_acc = val_confusion.totalValid*100
      print ("LAST  "..val_percent_acc_last.."  NEW  ".. val_percent_acc)
      val_percent_acc_last = maybe_save_model(model:get(custom_model_layer_index), epoch, opt.model_save_freq,
					      experiment_dir, val_percent_acc_last, val_percent_acc)
   end
end


---- Pseudo Label stuff ----


-- TODO extra data 
-- TODO unlabeled proportion
function pseudo_train_one_epoch(opt, trainData, extraData, unlabeledProportion, pseudo_loss_weight,
				optimState, model, criterion)
   model:training()

   local confusion = optim.ConfusionMatrix(10)
   local parameters, gradParameters = model:getParameters()

   -- labeled data
   
   DEBUG('labeled batch size: '..opt.batchSize)
   -- creates a random permutation of numbers 1 though the size of the training data,
   -- then splits them into batches.   
   local labeled_indices = torch.randperm(trainData.data:size(1)):long():split(opt.batchSize)
   -- remove last element so that all the batches have equal size
   if #labeled_indices > 1 then
      labeled_indices[#labeled_indices] = nil -- TODO huh?
   end
   DEBUG('Number of labeled batches: '.. #labeled_indices)

   local targets = nil
   if opt.no_cuda then
      targets = torch.FloatTensor(opt.batchSize)
   else
      targets = torch.CudaTensor(opt.batchSize)
   end

   local tic = torch.tic()  -- starts timer
   
   for t,v in ipairs(labeled_indices) do
      xlua.progress(t, #labeled_indices)
      
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
   -- If pseudo_loss_weight == 0, then the gradient update will be zero, so skip it.
   if pseudo_loss_weight > 0 then   
      DEBUG('unlabeled batch size: '..opt.unlabeledBatchSize)
      local unlabeled_indices = torch.randperm(extraData.data:size(1)):long():split(opt.unlabeledBatchSize)
      if #unlabeled_indices > 1 then
	 unlabeled_indices[#unlabeled_indices] = nil -- TODO huh?
      end
      DEBUG('Number of unlabeled batches: '.. #unlabeled_indices)
      
      for t,v in ipairs(unlabeled_indices) do
	 xlua.progress(t, #unlabeled_indices)
	 
	 local inputs = extraData.data:index(1, v)
	 
	 local feval = function(x)
	    if x ~= parameters then parameters:copy(x) end
	    gradParameters:zero() -- TODO huh? Doesn't look like this is used.
	    
	    local outputs = model:forward(inputs)
	    
	    -- compute the pseudo labels.
	    local _, targets = torch.max(outputs, 2)
	    if opt.no_cuda then
	       targets = targets:float()
	    else
	       targets = targets:cuda()
	    end
	       
	    local error = criterion:forward(outputs, targets) * pseudo_loss_weight
	     -- TODO I hope this is right...
	    local error_gradient = criterion:backward(outputs, targets) * pseudo_loss_weight
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

   pseudo_loss_weight = 0
   
   local val_percent_acc_last = 0
   for epoch = 1, opt.max_epoch do
      local epoch_debug_str = "==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']'
      DEBUG(epoch_debug_str)
      -- drop learning rate every "epoch_step" epochs
      if epoch % opt.epoch_step == 0 then
	 optimState.learningRate = optimState.learningRate/2
      end

      -- Update pseudo_loss_weight according to the schedule
      if epoch >= opt.pseudoStartingEpoch and epoch < opt.pseudoEndingEpoch then
	 pseudo_loss_weight = opt.maxPseudoLossWeight * (epoch - opt.pseudoStartingEpoch) / (opt.pseudoEndingEpoch - opt.pseudoStartingEpoch)
      elseif epoch >= opt.pseudoEndingEpoch then
	 pseudo_loss_weight = opt.maxPseudoLossWeight
      end
      DEBUG('pseudo loss weight: '..pseudo_loss_weight)
      
      local train_acc = pseudo_train_one_epoch(opt, provider.trainData, provider.extraData, unlabeledProportion, pseudo_loss_weight, 
					       optimState, model, criterion)
      val_confusion = evaluate_model(provider.valData, model)
      
      log_validation_stats(valLogger, model, epoch, train_acc, val_confusion,
			   optimState, experiment_dir)
      
      val_percent_acc = val_confusion.totalValid*100
      print ("LAST  "..val_percent_acc_last.."  NEW  ".. val_percent_acc)
      val_percent_acc_last = maybe_save_model(model:get(custom_model_layer_index), epoch, opt.model_save_freq,
					      experiment_dir, val_percent_acc_last, val_percent_acc)
   end
end

