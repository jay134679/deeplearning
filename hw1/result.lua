-- Homework 1: result.lua
-- Loads a trained MNIST model, and makes predictions on the test data in
-- 'mnist.t7/test_32x32.t7'. By default, it writes its predictions to a file
-- named predictions.csv, and loads the model from results/model.net.

require 'nn'
require 'optim'
require 'torch'

-- Parses the global 'arg' variable to get commandline arguments.
function parse_commandline()
   print "==> processing options"
   cmd = torch.CmdLine()
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
   cmd:text()
   options = cmd:parse(arg or {})
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

function prepare_test_data(data_filename, normalized_data_mean, normalized_data_std)
   print "==> loading test data"
   -- Load training and testing data. This will only use the testing data.
   loaded = torch.load(data_filename, 'ascii')
   test_data = {
      data = loaded.data:float(),
      labels = loaded.labels,
      size = loaded.data:size(1)
   }
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
   print "==> running model"
   -- classes
   classes = {'1','2','3','4','5','6','7','8','9','0'}
   -- This matrix records the current confusion across classes
   confusion = optim.ConfusionMatrix(classes)
   -- make predictions
   predictions_str = "Id,Prediction\n"

   for i = 1, test_data.size do
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

function write_predictions_csv(predictions_str, output_filename)
   print('==> saving ' .. output_filename .. '...')
   file = io.open(output_filename, "w")
   file:write(predictions_str)
   file:close()
   print('==> file saved')
end

function main()
   options = parse_commandline()
   model = torch.load(options.model_filename)
   test_data = prepare_test_data(options.data_filename, model.normalized_data_mean,
				 model.normalized_data_std)
   predictions_str = create_predictions_string(model, test_data)
   write_predictions_csv(predictions_str, options.output_filename)
end

main()

-- TODO why is it so accurate?
