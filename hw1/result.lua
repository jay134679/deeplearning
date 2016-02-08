-- Homework 1: result.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
--
-- This script loads a trained MNIST model, and makes predictions on
-- the test data in 'mnist.t7/test_32x32.t7'. By default, it writes
-- its predictions to a file named predictions.csv, and loads the
-- model from results/model.net.  NOTE: This script assumes that the
-- saved model has two extra keys: 'normalized_data_mean' and
-- 'normalized_data_std'.

require 'nn'
require 'optim'
require 'torch'

-- Parses the global 'arg' variable to get commandline arguments.
function parse_commandline()
   print "==> processing options"
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text("Homework 1 Results")
   cmd:text()
   cmd:text("Options:")
   cmd:option("-data_filename", "mnist.t7/test_32x32.t7",
	      "the name of the data on which to make predictions")
   cmd:option("-output_filename", "predictions.csv",
	      "the name of the CSV file that will contain the model's predictions.")
   cmd:option("-model_filename", "results/model.net",
	      "the name of the file that contains the trained model,")
   cmd:option("-num_data_to_test", -1, "The number of data points to test. If -1, defaults to the size of the test data.")
   cmd:text()
   local options = cmd:parse(arg or {})
   return options
end

-- Given a LongStorage, this finds the index with the greatest value.
-- Used to find the digit prediction given the results of nn.forward().
function max_index(row_storage)
   if #row_storage == 0 then
      return nil
   end
   local max_index = 1
   for i = 2,#row_storage do
      if row_storage[i] > row_storage[max_index] then
	 max_index = i
      end
   end
   return max_index
end

-- Loads the testing data from a file, normalizes it, and returns it.
function prepare_test_data(data_filename, num_data_to_test, normalized_data_mean, normalized_data_std)
   print "==> loading test data"
   -- Load training and testing data. This will only use the testing data.
   local loaded = torch.load(data_filename, 'ascii')

   local data_size = loaded.data:size(1)
   if num_data_to_test ~= -1 then
      print('==> setting the number of data points to test to: '.. num_data_to_test)
      data_size = num_data_to_test
   end
   
   local test_data = {
      data = loaded.data:float(),
      labels = loaded.labels,
      size = data_size
   }

   assert(normalized_data_mean ~= nil)
   assert(normalized_data_std ~= nil)
   print('==> normalizing mean to train data\'s: '..normalized_data_mean)
   print('==> normalizing std to train data\'s: '..normalized_data_std)
   
   -- normalize data using the training data's mean and standard deviation, as stored in the model.
   test_data.data[{ {},1,{},{} }]:add(-normalized_data_mean)
   test_data.data[{ {},1,{},{} }]:div(normalized_data_std)
   return test_data
end

-- Given the trained model and normalized test data, this evaluates the model
-- on each value of the test data. It writes its predictions to a
-- comma-delimited string, one prediction per line. It also prints the
-- confusion matrix.
function create_predictions_string(model, test_data)
   print("==> running model on test data with " .. test_data.size .. " entries.")
   model:evaluate()  -- Putting the model in evalate mode, in case it's needed.
   -- classes
   local classes = {'1','2','3','4','5','6','7','8','9','0'}
   -- This matrix records the current confusion across classes
   local confusion = optim.ConfusionMatrix(classes)
   -- make predictions
   local predictions_str = "Id,Prediction\n"

   for i = 1,test_data.size do
      -- get new sample
      local input = test_data.data[i]:double()
      local prediction_tensor = model:forward(input)
      local prediction = max_index(prediction_tensor:storage())
      confusion:add(prediction, test_data.labels[i])
      predictions_str = predictions_str .. i .. "," .. prediction .. "\n"
   end
   print(confusion)
   return predictions_str
end

-- Writes the given predictions string to the given output file.
function write_predictions_csv(predictions_str, output_filename)
   print('==> saving ' .. output_filename .. '...')
   local f = io.open(output_filename, "w")
   f:write(predictions_str)
   f:close()
   print('==> file saved')
end

function main()
   local options = parse_commandline()
   local model = torch.load(options.model_filename)
   local test_data = prepare_test_data(options.data_filename,
				       options.num_data_to_test,
				       model.normalized_data_mean,
				       model.normalized_data_std)
   local predictions_str = create_predictions_string(model, test_data)
   write_predictions_csv(predictions_str, options.output_filename)
end

main()
