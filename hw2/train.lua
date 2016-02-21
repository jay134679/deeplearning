-- Homework 2: train.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
-- Trainging and valdiation code.


require 'nn'
require 'optim'
require 'torch'
require 'xlua'
local c = require 'trepl.colorize'


function train_one_epoch(opt, trainData, optimState, model, criterion)
   model:training()

   local confusion = optim.ConfusionMatrix(10)
   
   local targets = torch.CudaTensor(opt.batchSize)
   -- TODO huh?
   local indices = torch.randperm(trainData.data:size(1)):long():split(opt.batchSize)
   -- remove last element so that all the batches have equal size
   indices[#indices] = nil

   local parameters, gradParameters = model:getParameters()
   
   local tic = torch.tic()  -- starts timer
   
   for t,v in ipairs(indices) do
      xlua.progress(t, #indices)
      
      local inputs = trainData.data:index(1, v) -- TODO huh?
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
      optim.sgd(feval, parameters, optimState) -- TODO it's dying here
   end
   
   confusion:updateValids()
   local time_secs = torch.toc(tic)
   DEBUG(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
	    confusion.totalValid * 100, time_secs))
   
   local train_acc = confusion.totalValid * 100
   return train_acc
end


function evaluate_model(opt, validateData, model, experiment_dir)
   model:evaluate()

   local confusion = optim.ConfusionMatrix(10)
   
   print(c.blue '==>'.." evaling")
   local bs = 25 -- TODO huh?
   for i = 1,provider.valData.data:size(1),bs do
      local outputs = model:forward(provider.valData.data:narrow(1, i, bs))
      confusion:batchAdd(outputs, provider.valData.labels:narrow(1, i, bs))
   end

   confusion:updateValids()
   DEBUG('val accuracy: '..confusion.totalValid * 100)
   return confusion
end


function log_validation_stats(valLogger, model, epoch, train_acc, val_confusion,
			      optimState, experiment_dir)
   local val_acc = val_confusion.totalValid * 100
   valLogger:add{train_acc, val_acc}
   valLogger:style{'-','-'}
   valLogger:plot()
   
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

-- NOTE: The main model MUST be the third layer.
function maybe_save_model(model, epoch, experiment_dir)
   -- save model every 5 epochs
   if epoch % 5 == 0 then
      local filename = paths.concat(experiment_dir, 'model.net')
      DEBUG('==> saving model to '..filename)
      torch.save(filename, model:get(3))
   end
end


-- This modifies model and val_logger.
-- This returns the percentage of samples that were correctly
-- classified on the validation set and the average number of
-- milliseconds per sample required to train the model.
function train_validate_max_epochs(opt, provider, model, experiment_dir)
   local valLogger = optim.Logger(paths.concat(experiment_dir, 'val.log'))
   valLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
   valLogger.showPlot = false

   print(c.blue'==>' ..' setting criterion')
   local criterion = nn.CrossEntropyCriterion():cuda()

   print(c.blue'==>' ..' configuring optimizer')   
   local optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = opt.learningRateDecay,
   }

   for epoch = 1,opt.max_epoch do
      local epoch_debug_str = "==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']'
      DEBUG(epoch_debug_str)
      print(epoch_debug_str)
      -- drop learning rate every "epoch_step" epochs
      if epoch % opt.epoch_step == 0 then
	 optimState.learningRate = optimState.learningRate/2
      end
      
      local train_acc = train_one_epoch(opt, provider.trainData, optimState,
					model, criterion)
      local val_confusion = evaluate_model(opt, validateData, model)
      
      log_validation_stats(valLogger, model, epoch, train_acc, val_confusion,
			   optimState, experiment_dir)
      -- TODO save model if performs better? Jake doesn't...
      maybe_save_model(model, epoch, experiment_dir)
   end
end
