-- Homework 1: test.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods


-- returns a confusion matrix
-- modifies logger, modifies model, but only to put it in evaluation mode and call forward.
function evaluate_model(opt, eval_data, model, logger)
   -- local vars
   local time = sys.clock()
   
   -- This matrix records the current confusion across classes
   local classes = {'1','2','3','4','5','6','7','8','9','0'}
   local confusion = optim.ConfusionMatrix(classes)

   -- Retrieve parameters
   -- this extracts and flattens all the trainable parameters of the model
   -- into a 1-dim vector
   local parameters, _ = model:getParameters()

   -- TODO 'average' is a global defined train().
   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> evaluating data set:')
   for t = 1,eval_data:size() do
      -- disp progress
      xlua.progress(t, eval_data:size())

      -- get new sample
      local input = eval_data.data[t]:double()
      local target = eval_data.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / eval_data:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   logger:add{['% mean class accuracy'] = confusion.totalValid * 100}
   if opt.plot then
      logger:style{['% mean class accuracy'] = '-'}
      logger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   return confusion
end





