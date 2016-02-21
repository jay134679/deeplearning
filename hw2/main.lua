-- Homework 2: main.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
-- Trains, validates, and tests data for homework 2.

require 'torch'
local c = require 'trepl.colorize'

-- Locally defined files
require 'augment_data'
require 'exp_setup'
require 'provider'
require 'train'

function parse_cmdline()
   local opt = lapp[[
      --size                  (default "tiny")      size of data to use: tiny, small, full.
      --exp_name              (default "")          name of the current experiment. optional.
      --results_dir           (default "results")   directory to save results
      --debug_log_filename    (default "debug.log")  filename of debugging output
      -b,--batchSize          (default 64)          batch size
      -r,--learningRate       (default 1)        learning rate
      --learningRateDecay     (default 1e-7)      learning rate decay
      --weightDecay           (default 0.0005)      weightDecay
      -m,--momentum           (default 0.9)         momentum
      --epoch_step            (default 25)          reduce learning rate every this many epochs
      --max_epoch             (default 300)         maximum number of iterations
      --model                 (default vgg_bn_drop) model name
      --backend               (default nn)          backend, nn or cudnn
      --use_cuda              (default true)        whether to use cuda or not
   ]]
   return opt
end

function load_provider(size)
   print(c.blue '==>' ..' loading data')
   -- TODO delete provider.t7 this file once you add unlabeled data to provider.lua.
   data_filename = 'provider.'..size..'.t7'
   data_file = io.open(data_filename, 'r')
   provider = nil
   if data_file ~= nil then
      DEBUG('loading data from file...')
      print('loading data from file...')
      provider = torch.load(data_filename)
   else
      DEBUG('downloading data...')
      print('downloading data...')
      provider = Provider(size)
      provider:normalize()
      provider.trainData.data = provider.trainData.data:float()
      provider.valData.data = provider.valData.data:float()
      torch.save(data_filename, provider)
   end
   return provider
end

-- returns the constructed sequential model, and the index of the sub-model from
-- the models/ directory.
function load_model(model_name, use_cuda)
   DEBUG('loading model: '..model_name)
   DEBUG('use_cuda: '..tostring(use_cuda))
   
   local model = nn.Sequential()
   -- 1st layer: data augmentation
   add_batch_flip(model)

   custom_model_layer_index = nil

   if use_cuda then
      require 'cunn'
      model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
      model:add(dofile('models/'..model_name..'.lua'):cuda())
      model:get(2).updateGradInput = function(input) return end
      custom_model_layer_index = 3

      -- TODO will we ever have access to cudnn?
      if opt.backend == 'cudnn' then
	 require 'cudnn'
	 cudnn.convert(custom_model_layer_index, cudnn)
      end
   else
      model:add(dofile('models/'..model_name..'.lua'))
      custom_model_layer_index = 2
   end   
   DEBUG(model)
   return model, custom_model_layer_index
end

function main()
   opt = parse_cmdline()
   experiment_dir = setup_experiment(opt)
   -- DEBUG function now callable
   provider = load_provider(opt.size)
   model, custom_model_layer_index = load_model(opt.model, opt.use_cuda)
   train_validate_max_epochs(opt, provider, model, custom_model_layer_index, experiment_dir)
   print('Experiment complete.')
end

main()
