-- Homework 1: result.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
--
-- This script loads a trained MNIST model, and makes predictions on
-- the test data in 'mnist.t7/test_32x32.t7'.
-- The user must specify the model file name via the 'model_filename' flag.
-- By default, it writes its predictions to predictions.csv.
--
-- IMPORTANT NOTE
-- This script relies on the 'prepare_data.lua' file written
-- for this assignment. They must be in lua's file lookup path in order for this
-- script to run.

-- Example usage:
-- Example 1: Load model in results/mymodel.net, and write the output to
-- results/predictions.csv:
--
-- th result.lua -model_filename results/mymodel.net

-- Example 2: Only load the 'small' test data test, load model in
-- results/mymodel.net, and write the output to predictions.csv:
--
-- th result.lua -size small -model_filename results/mymodel.net -output_filename results/myresults.log

require 'nn'
require 'optim'
require 'torch'

-- our custom code. This must be in the same directory.
require 'prepare_data'


-- Parses the global 'arg' variable to get commandline arguments.
function parse_commandline()
   print "==> processing options"
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text("Homework 1 Results")
   cmd:text()
   cmd:text("Options:")
   cmd:option('-size', 'full', 'how many samples do we load from test data: tiny | small | full. Required.')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option("-output_filename", "predictions.csv",
	      "the name of the CSV file that will contain the model's predictions. Required")
   cmd:option("-model_filename", "",
	      "the name of the file that contains the trained model. Required!")
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


-- Given the trained model and normalized test data, this evaluates the model
-- on each value of the test data. It writes its predictions to a
-- comma-delimited string, one prediction per line. It also prints the
-- confusion matrix.
function create_predictions_string(model, test_data)
   print("==> running model on test data with " .. test_data:size() .. " entries.")
   model:evaluate()  -- Putting the model in evalate mode, in case it's needed.
   -- classes
   local classes = {'1','2','3','4','5','6','7','8','9','0'}
   -- This matrix records the current confusion across classes
   local confusion = optim.ConfusionMatrix(classes)
   -- make predictions
   local predictions_str = "Id,Prediction\n"

   for i = 1,test_data:size() do
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


-- This is the function that runs the script. It checks the command line flags
-- and executes the program.
function main()
   local options = parse_commandline()
   if options.model_filename == '' then
      print 'ERROR: You must set -model_filename'
      exit()
   end
   if options.output_filename == ''  then
      print 'ERROR: You must set -output_filename'
      exit()
   end
   if options.size == ''  then
      print 'ERROR: You must set -size: full | small | tiny'
      exit()
   end
   
   local dummy_frac = 0.75
   local _, _, testData = build_datasets(options.size, dummy_frac)
   local model = torch.load(options.model_filename)
      
   local predictions_str = create_predictions_string(model, testData)
   write_predictions_csv(predictions_str, options.output_filename)
end


main()
