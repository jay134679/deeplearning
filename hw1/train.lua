-- Homework 1: train.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods


function choose_optim_method(opt)
   print '==> configuring optimizer'

   if opt.optimization == 'CG' then
      optimState = {
	 maxIter = opt.maxIter
      }
      optimMethod = optim.cg

   elseif opt.optimization == 'LBFGS' then
      optimState = {
	 learningRate = opt.learningRate,
	 maxIter = opt.maxIter,
	 nCorrection = 10
      }
      optimMethod = optim.lbfgs

   elseif opt.optimization == 'SGD' then
      optimState = {
	 learningRate = opt.learningRate,
	 weightDecay = opt.weightDecay,
	 momentum = opt.momentum,
	 learningRateDecay = 1e-7
      }
      optimMethod = optim.sgd

   elseif opt.optimization == 'ASGD' then
      optimState = {
	 eta0 = opt.learningRate,
	 t0 = trsize * opt.t0
      }
      optimMethod = optim.asgd

   else
      error('unknown optimization method')
   end
   return optimMethod, optimState
end


-- This modifies model, returns percent accuracy and time in ms.
function train_one_epoch(opt, trainData, optimMethod, optimState, model, criterion)
   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- Retrieve parameters and gradients:
   -- this extracts and flattens all the trainable parameters of the model
   -- into a 1-dim vector
   parameters, gradParameters = model:getParameters()

   -- This matrix records the current confusion across classes
   local classes = {'1','2','3','4','5','6','7','8','9','0'}
   local confusion = optim.ConfusionMatrix(classes)
   
   -- shuffle at each epoch
   local shuffle = torch.randperm(trainData:size())

   -- do one epoch
   print('==> doing epoch on training data:')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())
      
      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]:double()
         local target = trainData.labels[shuffle[i]]
         table.insert(inputs, input)
         table.insert(targets, target)
      end
      -- create closure to evaluate f(X) and df/dX
      -- This modifies 'model' and 'confusion'.
      local feval = function(x)
	 -- get new parameters
	 if x ~= parameters then
	    parameters:copy(x)
	 end
	 
	 -- reset gradients
	 gradParameters:zero()
	 
	 -- f is the average of all criterions
	 local f = 0
	 
	 -- evaluate function for complete mini batch
	 for i = 1,#inputs do
	    -- estimate f
	    local output = model:forward(inputs[i])
	    local err = criterion:forward(output, targets[i])
	    f = f + err
	    
	    -- estimate df/dW
	    local df_do = criterion:backward(output, targets[i])
	    model:backward(inputs[i], df_do)
	    
	    -- update confusion
	    confusion:add(output, targets[i])
	 end
	 
	 -- normalize gradients and f(X)
	 gradParameters:div(#inputs)
	 f = f/#inputs
	 
	 -- return f and df/dX
	 return f,gradParameters
      end  -- end closure

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
	 -- TODO 'average' is a global used in 5_test.lua. Not sure
	 -- what it is, looks like it gets rewritten each iteration
	 -- here.
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   time_ms = time*1000
   print("\n==> time to learn 1 sample = " .. (time_ms) .. 'ms')

   percent_valid = confusion.totalValid * 100
   -- print(confusion)

   -- TODO
   -- --last_global_avg = confusion.totalValid
   -- -- update logger/plot
   -- logger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   -- if opt.plot then
   --    logger:style{['% mean class accuracy (train set)'] = '-'}
   --    logger:plot()
   -- end
   return percent_valid, time_ms
end




