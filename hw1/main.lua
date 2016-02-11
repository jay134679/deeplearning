-- Homework 1: main.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
-- Trains, validates, and tests data for homework 1.

-- local libs
require 'prepare_data'
require 'prepare_model'
require 'train'
require 'test'

-- global libs
require 'torch'

----------------------------------------------------------------------
print '==> processing options'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- data:
cmd:option('-size', 'tiny', 'how many samples do we load: tiny | small | full')
cmd:option('-tr_frac', 0.75, 'fraction of original train data assigned to validation ')
-- model:
cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot') -- TODO
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-batchSizeArray', {1,10,50,100}, 'batch sizes to try')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'double', 'type: double | float | cuda')

-- NEW flags!
cmd:option('-maxEpoch', 10, 'maximum number of epochs to train')
cmd:option('-experimentName', '', 'The name of the experiment. Used to name the log files. Defaults to opt.mode')
cmd:option('-mode', 'prod', 'chooses what program to run. options: prod | batch_size')


cmd:text()
local opt = cmd:parse(arg or {})

----------------------------------------------------------------------
print '==> training!'

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)


function savemodel(model, filename)
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
end

-- defines global loggers
-- TODO get rid of these
function start_logging()
   valLogger = optim.Logger(paths.concat(opt.save, 'validate'..opt.batchSize..'.log'))  
   testLogger = optim.Logger(paths.concat(opt.save, 'test'..opt.batchSize..'.log'))
end


-- This modifies model and val_logger.
-- This returns the percentage of samples that were correctly
-- classified on the validation set and the average number of
-- milliseconds per sample required to train the model.
function train_validate_max_epochs(opt, trainData, validateData,
				   model, criterion, output_filename,
				   val_logger)
   print '==> defining some tools'
   optimMethod, optimState = choose_optim_method(opt)

   avg_time_ms = 0.0
   local best_val_percent_valid = 0.0
   for epoch = 1,opt.maxEpoch do
      print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
      -- train.lua
      _, time_ms = train_one_epoch(opt, trainData, optimMethod, optimState, model, criterion)
      avg_time_ms = avg_time_ms + time_ms
      
      -- run evaluation on validation dataset
      local val_confusion = evaluate_model(opt, validateData, model, val_logger)
      local val_percent_valid = val_confusion.totalValid * 100

      -- save model if performs better
      print("previous best Accuracy :::: " .. best_val_percent_valid .. "   NEW Accuracy ... ".. val_percent_valid)
      -- only save the new version if it's noticably better
      if val_percent_valid - best_val_percent_valid > 0.1 then
	 print('new model is better, saving it')
	 savemodel(model, output_filename)
	 best_val_percent_valid = val_percent_valid
      end
   end
   avg_time_ms = avg_time_ms / opt.maxEpoch
   -- test.lua
   -- local val_confusion = evaluate_model(opt, validateData, model, val_logger)
   -- local val_percent_valid = val_confusion.totalValid * 100
   
   return val_percent_valid, avg_time_ms
end

-- This function goes through the opt.batchSizeArray, and trains a model using
-- each of its values as the training mini-batch size. The accuracy and training
-- time for each model is saved in Logger objects.
function change_batch_size()
    -- prepare_data.lua
   local trainData, validateData, testData = build_datasets(opt.size, opt.tr_frac)

   local timestamp = os.date("%m%d%H%M%S")
   local accuracy_logger = optim.Logger(paths.concat(opt.save, 'val_accuracy.'..timestamp..'.log'))
   local train_time_logger = optim.Logger(paths.concat(opt.save, 'train_time.'..timestamp..'.log'))

   -- experiment with difference batch sizes
   for i = 1, #opt.batchSizeArray do
      -- prepare_model.lua
      local model = build_model(opt.model, trainData.mean, trainData.std)
      local criterion = build_criterion(opt.loss, trainData, validateData, testData, model)

      -- set specific batchsize for expirement
      opt.batchSize = opt.batchSizeArray[i]
      local output_filename = paths.concat(opt.save, 'model'..opt.batchSize..'.net')
      -- change save path to folder for specific batchsize
      start_logging()
      -- train and run validation
      local val_percent_valid, avg_train_time_ms =
	 train_validate_max_epochs(opt, trainData, validateData, model,
				   criterion, output_filename, valLogger)
      -- log the validation accuracy and average training time for this batch size
      accuracy_logger:add{["percent correct"] = val_percent_valid}
      train_time_logger:add{["average time per epoch per sample"] = avg_train_time_ms}
      
      -- see how the model does on the test data. Don't save the confusion matrix, just print it.
      print('\n\n')
      print('Test (not validation!) data performance')
      evaluate_model(opt, testData, model, testLogger)
   end

   if opt.plot then
      accuracy_logger:style{["percent correct"] = '-'}
      accuracy_logger:plot()
      train_time_logger:style{["average time per epoch per sample"] = '-'}
      train_time_logger:plot()
   end   
end


-- Simply loads the data, trains the model until opts.maxEpochs, checks validation set accuracy, checks test set accuracy.
function train_validate_save_model()

    -- set up loggers
   local timestamp = os.date("%m%d%H%M%S")
   local val_accuracy_logger = optim.Logger(paths.concat(opt.save, opt.experimentName..'.val_accuracy.'..timestamp..'.log'))
   local test_accuracy_logger = optim.Logger(paths.concat(opt.save, opt.experimentName..'.test_accuracy.'..timestamp..'.log'))

    -- prepare file to save model
    output_filename = paths.concat(opt.save, opt.experimentName..'.model.'..timestamp..'.net')

    -- prepare_data.lua
   local trainData, validateData, testData = build_datasets(opt.size, opt.tr_frac)

   -- build model and criterion
   local model = build_model(opt.model, trainData.mean, trainData.std)
   local criterion = build_criterion(opt.loss, trainData, validateData, testData, model)

   local val_percent_valid, avg_train_time_ms =
      train_validate_max_epochs(opt, trainData, validateData, model,
				criterion, output_filename, val_accuracy_logger)

   -- print how the model does on the test data.
   print('\n\n')
   print('Test (not validation!) data performance')
   evaluate_model(opt, testData, model, test_accuracy_logger)

   -- save the final model
   --filename = paths.concat(opt.save, opt.experimentName..'.model.'..timestamp..'.net')
   --savemodel(model, output_filename)  
end


-- Example usage:
-- th main.lua -mode prod -size tiny -maxEpoch 3
function main()
   -- set filename
   --filename = paths.concat(opt.save, opt.experimentName..'.model.'..timestamp..'.net')

   if opt.experimentName == '' then
      opt.experimentName = opt.mode
   end
   
   if opt.mode == 'prod' then
      train_validate_save_model()
   --[[elseif opt.mode == 'batch_size' then
      change_batch_size()]]
   -- TODO add more modes here.
   else
      print('no running mode chosen! Set the -mode flag')
   end
end


main()
