-- Homework 2: result.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)


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

-- TODO give --model_filename the name of our final model as a defualt.

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
   cmd:option('--size', 'full', 'how many samples do we load from test data: tiny | small | full. Required.')
   cmd:option('--output_dir', '', 'subdirectory to save/log experiments in. Required.')
   cmd:option("--output_filename", "predictions.csv",
	      "the name of the CSV file that will contain the model's predictions. Required")
   cmd:option("--model_filename", '',
	      "the name of the file that contains the trained model. Required!")
   cmd:option("--num_data_to_test", -1, "The number of data points to test. If -1, defaults to the size of the test data.")
   cmd:text()
   local options = cmd:parse(arg or {})   
   return options
end

-- Given the trained model and normalized test data, this evaluates the model
-- on each value of the test data. It writes its predictions to a
-- comma-delimited string, one prediction per line. It also prints the
-- confusion matrix.
function create_predictions_string(model, testData)
   print("==> running model on test data with " .. testData:size() .. " entries.")
   model:evaluate()  -- Putting the model in evalate mode, in case it's needed.

   local classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
   -- This matrix records the current confusion across classes
   local confusion = optim.ConfusionMatrix(classes)

   local predictions_str = "Id,Prediction\n"

   print('==> evaluating')
   local batch_size = 25
   for i = 1, testData.data:size(1), batch_size do
      xlua.progress(i+batch_size, testData.data:size(1))

      local outputs = model:forward(testData.data:narrow(1, i, batch_size):cuda())
      confusion:batchAdd(outputs, testData.labels:narrow(1, i, batch_size))
      for j = 1,batch_size do
         -- This call gets the index of the largest value in the outputs[j] list.
         local value, index = outputs[j]:topk(1, 1, true)
         prediction = assert(index[1]) -- index is a Tensor, this makes it a number
	 labelNum = i - 1 + j
         predictions_str = predictions_str .. labelNum .. "," .. tostring(prediction) .. "\n"
      end
   end
   confusion:updateValids()
   print(confusion)
   return predictions_str
end


-- Writes the given predictions string to the given output file.
function write_predictions_csv(predictions_str, output_dir, output_filename)
   local output_filepath = paths.concat(output_dir, output_filename)
   print('==> saving ' .. output_filepath .. '...')
   local f = io.open(output_filepath, "w")
   f:write(predictions_str)
   f:close()
   print('==> file saved')
end


function run(size, model_filename, output_dir, output_filename)
   -- NOTE: This are global on purpose, so this can be tested in the REPL.
   provider = load_provider(size, 'evaluate', false)

   model = torch.load(model_filename):cuda()
   local predictions_str = create_predictions_string(model, provider.testData)
   write_predictions_csv(predictions_str, output_dir, output_filename)
end

-- This is the function that runs the script. It checks the command line flags
-- and executes the program.
function main()
   local options = parse_commandline()
   if options.model_filename == '' then
      print 'ERROR: You must set --model_filename'
      exit()
   end
   if options.output_dir == '' then
      print 'ERROR: You must set --output_dir'
      exit()
   end
   if options.output_filename == '' then
      print 'ERROR: You must set --output_filename'
      exit()
   end
   if options.size ~= 'full' and options.size ~= 'small' and options.size ~= 'tiny' then
      print 'ERROR: You must set -size: full | small | tiny'
      exit()
   end
   
   run(options.size, options.model_filename, options.output_dir, options.output_filename)
end


main()
