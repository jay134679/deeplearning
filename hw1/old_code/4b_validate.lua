----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining validate procedure'

-- validate function
function validate(valLogger, ModelUpdateLogger)
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on validate set:')
   for t = 1,validateData:size() do
      -- disp progress
      xlua.progress(t, validateData:size())

      -- get new sample
      local input = validateData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = validateData.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / validateData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   assert(mean)
   assert(std)
   savemodel(confusion, ModelUpdateLogger, mean, std)

   -- update log/plot
   valLogger:add{['% mean class accuracy (validation set)'] = confusion.totalValid * 100}
   if opt.plot then
      valLogger:style{['% mean class accuracy (validation set)'] = '-'}
      valLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end

-- the mean and std from the training data must be saved with the model, so they
-- can be used to normalize the test data.
function savemodel(confusion, ModelUpdateLogger, normalized_data_mean, normalized_data_std)
   new_accuracy = confusion.totalValid
   table.insert(accuracy_tracker, new_accuracy)
   if new_accuracy-old_accuracy > epsilon  then
      --ModelUpdateLogger:add{['Model Updated ==> % mean class accuracy (validation set)'] = new_accuracy}
      ModelUpdateLogger:add{epoch-1, new_accuracy}
      -- Saving the normalized mean and standard deviation to prepare testing data.
      model['normalized_data_mean'] = normalized_data_mean
      model['normalized_data_std'] = normalized_data_std
      -- save/log current net
      local filename = paths.concat(opt.save, 'model'..opt.batchSize..'.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('New model is better ==> saving model to '..filename)
      torch.save(filename, model)
      old_accuracy=new_accuracy
   end
end
