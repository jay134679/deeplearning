-- Homework 2: main.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
-- Trains, validates, and tests data for homework 2.

require 'torch'
local c = require 'trepl.colorize'

-- Locally defined files
require 'augment_data'
require 'exp_setup'
require 'provider'

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
      --epoch_step            (default 25)          epoch step
      --max_epoch             (default 300)           maximum number of iterations
      --model                 (default vgg_bn_drop)     model name
      --backend               (default nn)            backend, nn or cudnn
   ]]
   return opt
end

function load_provider()
   print(c.blue '==>' ..' loading data')
   -- TODO delete provider.t7 this file once you add unlabeled data to provider.lua.
   data_file = io.open('provider.t7', 'r')
   if data_file ~= nil then
      DEBUG('loading data from file...')
      provider = torch.load('provider.t7')
   else
      DEBUG('downloading data...')
      provider = Provider()
      provider:normalize()
      -- TODO does the 'float' call have to be changed in cuda mode?
      -- Jake leaves them as float in his cuda code...
      provider.trainData.data = provider.trainData.data:float()
      provider.valData.data = provider.valData.data:float()
      torch.save('provider.t7', provider)
   end
   return provider
end


-- NOTE: The main model MUST be the third thing. Validation asssumes it is.
function load_model(model_name)
   local model = nn.Sequential()
   add_batch_flip(model) -- TODO 
   model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
   -- NOTE: This layer must be the third one!
   model:add(dofile('models/'..model_name..'.lua'):cuda())
   model:get(2).updateGradInput = function(input) return end

   -- TODO will we ever have access to cudnn?
   if opt.backend == 'cudnn' then
      require 'cudnn'
      cudnn.convert(model:get(3), cudnn)
   end
   DEBUG('loading model...')
   DEBUG(model)
   return model
end

function main()
   opt = parse_cmdline()
   experiment_dir = setup_experiment(opt)
   -- DEBUG function now callable
   provider = load_provider(opt.size)
   model = load_model(opt.model)
   train_validate_max_epochs(opt, provider, model, experiment_dir)
   print('Experiment complete.')
end

main()
