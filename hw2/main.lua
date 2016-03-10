-- Homework 2: main.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
-- Trains, validates, and tests data for homework 2.

require 'torch'

-- Locally defined files
require 'augment_data'
require 'exp_setup'
require 'provider'
require 'train'

function parse_cmdline()
   local opt = lapp[[
      --size                  (default "tiny")      size of data to use: tiny, small, full.
      --exp_name              (default "")          name of the current experiment. optional.
      --epoch_step            (default 25)          reduce learning rate every x epochs.
      --max_epoch             (default 300)         maximum number of iterations.
      --model_save_freq       (default 50)          save the model every x epochs.
      --model                 (default vgg_bn_drop) model name
      --backend               (default nn)          backend, nn or cudnn
      --no_cuda                                     whether to use cuda or not. defaults to false, so cuda is used by default.
      --results_dir           (default "results")   directory to save results
      --debug_log_filename    (default "debug.log")  filename of debugging output
      -b,--batchSize          (default 64)          batch size
      -r,--learningRate       (default 1)           learning rate
      --learningRateDecay     (default 1e-7)        learning rate decay
      --weightDecay           (default 0.0005)      weightDecay
      -m,--momentum           (default 0.9)         momentum
      -a, --augmented                               defaults to false
      --pre_train_res         (default "")          file for pre train
      --usePseudoLabels                             If true, use pseudo label training method.
      --unlabeledBatchSize    (default 64)          Batch size for pseudo-labeled unlabeled data.
      --maxPseudoLossWeight   (default 3)           The maximum weight given to the loss of the pseudo-labeled data .
      --pseudoStartingEpoch   (default 100)         pseudo label loss weight is zero until this epoch. Adjust according to max_epoch.
      --pseudoEndingEpoch     (default 250)         psudeo label loss max weight realized at this epoch. Adjust according to max_epoch.
   ]]
   return opt
end

-- returns the constructed sequential model, and the index of the sub-model from
-- the models/ directory.
function load_model(model_name, no_cuda)
   DEBUG('loading model: '..model_name)
   DEBUG('using CUDA: '..tostring(not no_cuda))
   
   local model = nn.Sequential()
   -- 1st layer: data augmentation
   add_batch_flip(model)  -- TODO TODO TODO incorporate this into the data augmentation!
 
   custom_model_layer_index = nil
   if no_cuda then
      model:add(dofile('models/'..model_name..'.lua'))
      custom_model_layer_index = 2
   else
      require 'cunn'
      model:get(1) -- Maybe here? model:get(1).weight:copy(a)
      model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
      model:add(dofile('models/'..model_name..'.lua'):cuda())
      model:get(2).updateGradInput = function(input) return end
      custom_model_layer_index = 3
      -- TODO will we ever have access to cudnn?
      if opt.backend == 'cudnn' then
	 require 'cudnn'
	 cudnn.convert(custom_model_layer_index, cudnn)
      end
   end   
   DEBUG(model)
   return model, custom_model_layer_index
end

function kMeansPreTraining(pre_train_res, model)
   if pre_train_res ~= "" then
      DEBUG('Pre-training model with K-means...')
      centroids = torch.load(opt.pre_train_res)
      reshaped = torch.reshape(centroids, 64,3,3,3)
      model:get(3):get(1).weight:copy(reshaped)
   end
end   

function main()
   opt = parse_cmdline()
   experiment_dir = setup_experiment(opt)
   -- DEBUG function now callable
   print(opt)

   --[[if opt.pre_train_res ~= "" then
      print ("==> using centroids to initialize filters")
      centroids = torch.load(opt.pre_train_res)
      reshaped = torch.reshape(centroids, 64,3,3,3)
      model:get(3):get(1).weight:copy(reshaped)]]
   if opt.usePseudoLabels then
      provider = load_provider(opt.size, 'unlabeled', opt.augmented, true)
      model, custom_model_layer_index = load_model(opt.model, opt.no_cuda)
      kMeansPreTraining(opt.pre_train_res, model)
      pseudo_train_validate_max_epochs(opt, provider, model, custom_model_layer_index, experiment_dir)
   else
      provider = load_provider(opt.size, 'training', opt.augmented, true)
      model, custom_model_layer_index = load_model(opt.model, opt.no_cuda)
      kMeansPreTraining(opt.pre_train_res, model)
      train_validate_max_epochs(opt, provider, model, custom_model_layer_index, experiment_dir)
   end
   print('Experiment complete.')
end

main()






