-- Homework 2: result.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)

-- TODO THIS DOES NOT WORK!!! SO SLOW! test in an interactive shell doofus.
-- TODO get this to work on CUDA. it's too slow.

-- TODO SET DEFAULT VALUE FOR --model_filename so they can run without flags!


-- This script loads a trained STL-10 model, and makes predictions on
-- the test data in 'stl-10/test.t7b'
-- The user must specify the model file name via the 'model_filename' flag.
-- By default, it writes its predictions to predictions.csv.
--
-- IMPORTANT NOTE
-- This script relies on the 'provider.lua' file written
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
require 'cunn'
-- our custom code. This must be in the same directory.
require 'provider'


-- Parses the global 'arg' variable to get commandline arguments.
function parse_commandline()
   print "==> processing options"
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text("Homework 2 Results")
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
   local classes = {}
   for i = 1,100 do
      classes[i] = tostring(i)
   end

   -- This matrix records the current confusion across classes
   local confusion = optim.ConfusionMatrix(classes)

   local predictions_str = "Id,Prediction\n"

   model:float() -- TODO without this it fails, but it's so slow.
   print('==> evaluating')
   local batch_size = 25
   for i = 1,test_data.data:size(1),batch_size do
      local outputs = model:forward(test_data.data:narrow(1, i, batch_size))
      confusion:batchAdd(outputs, test_data.labels:narrow(1, i, batch_size))
      for j = 1,batch_size do
         prediction = max_index(outputs[j]:storage())
         predictions_str = predictions_str .. j .. "," .. prediction .. "\n"
      end
   end
   confusion:updateValids()
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
   if options.output_filename == '' then
      print 'ERROR: You must set -output_filename'
      exit()
   end
   if options.size ~= 'full' and options.size ~= 'small' and options.size ~= 'tiny' then
      print 'ERROR: You must set -size: full | small | tiny'
      exit()
   end
   
   local provider = Provider(options.size)
   provider:normalize()
   local model = torch.load(options.model_filename):cuda() -- TODO is this needed/desirable?
   local predictions_str = create_predictions_string(model, provider.testData)
   write_predictions_csv(predictions_str, options.output_filename)
end


main()
