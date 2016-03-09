-- Homework 2: tsne.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
-- Creates a t-SNE tensor for a given trained model's layer number.

require 'cunn'
require 'nn'
require 'torch'

require 'provider'


function parse_cmdline()
   local options = lapp[[
      --layer_number   (default 8)           The layer of the model to extract: 8, 16, 28, or 40.
      --model_dir      (default "")          The directory with the model file to load, and where the output file will go.
      --model_name     (default "model.net") The filename of the model file.
      --num_data       (default 1000)        The number of data points to use.
   ]]
   return options
end


function loadData(numData)
   local provider = load_provider('full', 'evaluate', false)

   local testData = provider.testData
   -- don't normalize, tnse does that already
   
   print('Test data before reshaping')
   print(testData)
   
   testData.size = function() return numData end
   testData.data = testData.data:narrow(1, 1, numData):cuda()
   testData.labels = testData.labels:narrow(1, 1, numData):float()
   
   print('Test data after narrowing')
   print(testData)

   return testData
end

function loadModel(modelDir, modelName)
   local modelFilename = paths.concat(modelDir, modelName)
   print('Loading model: '..modelFilename)
   local model = torch.load(modelFilename):cuda()
   -- TODO clear the model and gc if the program OOMs in main memory (not gpu).
   model:evaluate()
   print 'model:'
   print(model)
   return model
end

function runModel(model, dataObj, layerNumber)
   print('Extracting model layer '..layerNumber)
   local layer = model:get(layerNumber)
   print(layer)
   print('Running forward pass...')
   -- Batching data, otherwise it OOMs.
   local batchSize = 25
   local layerOutputs = nil
   for i = 1, testData.data:size(1), batchSize do
      xlua.progress(i+batchSize, dataObj.data:size(1))
      local unusedOutputs = model:forward(dataObj.data:narrow(1, i, batchSize):cuda())   
      if layerOutputs then
	 layerOutputs:cat(layer.output, 1)
      else
	 layerOutputs = torch.DoubleTensor(layer.output:size()):copy(layer.output)
      end
   end
   print('Layer output dimensions:')
   print(layerOutputs:size())
   return layerOutput
end

function runTsne(layerOutput)
   print('Preparing data for t-SNE...')
   local x = torch.DoubleTensor(layerOutput:size()):copy(layerOutput)
   -- Flatten data
   x:resize(x:size(1), x:size(2) * x:size(3) * x:size(4))
   print('Flattened data dimensions:')
   print(x:size())

   local m = require 'manifold';
   local opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
   print('t-SNE options:')
   print(opts)
   print('Running t-SNE...')
   local mapped_x1 = m.embedding.tsne(x, opts)
   print('t-SNE complete!')
   im_size = 4096
   print('Calling draw_image_map...')
   local map_im = m.draw_image_map(mapped_x1, 
                                   x:resize(layerOutput:size(1), layerOutput:size(2), layerOutput:size(3), layerOutput:size(4)),
                                   im_size, 0, true)
   return map_im
end
   
function saveImage(tsneTensor, outputDir, layerNumber, numData)
   local outputFilename = 'tnse.layer'..layerNumber..'.ndata'..numData..'.t7'
   local outputPath = paths.concat(outputDir, outputFilename)
   print('Saving file: '..outputPath)
   torch.save(outputPath, tsneTensor)
end

function main()
   local options = parse_cmdline()
   if options.model_dir == '' then
      print 'ERROR: You must set --model_dir'
      exit()
   end

   local dataObj = loadData(options.num_data)
   local model = loadModel(options.model_dir, options.model_name)
   local layerOutput = runModel(model, dataObj, options.layer_number)
   local tsneTensor = runTsne(layerOutput)
   saveImage(tsneTensor, options.model_dir, options.layer_number, options.num_data)
end

main()
