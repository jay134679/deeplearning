-- main.lua
-- Trains, validates, and tests data for homework 1.
-- Maya Rotmensch (mer567) and Alex Pine (akp258)

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
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-batchSizeArray', {1})--TODO{1,10,50,100}, 'batch sizes to try')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
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


-- TODO put these in opt?
-- global constants
EPSILON = 0.000001
MAX_EPOCHS = 15

-- saves the model to disk is new_accuracy is larger than old_accuracy by at least EPSILON.
function savemodel(model, filename, new_accuracy, old_accuracy, epoch, logger)
   if new_accuracy - old_accuracy > EPSILON  then
      --logger:add{['Model Updated ==> % mean class accuracy (validation set)'] = new_accuracy}
      logger:add{epoch-1, new_accuracy}
      -- save/log current net
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('New model is better ==> saving model to '..filename)
      torch.save(filename, model)
   end
end

-- defines global loggers
-- TODO de-globalize
function start_logging()
   --train_new_name = 'train'..opt.batchSize..'.log'
   --print (train_new_name)
   trainLogger = optim.Logger(paths.concat(opt.save, 'train'..opt.batchSize..'.log'))
   valLogger = optim.Logger(paths.concat(opt.save, 'validate'..opt.batchSize..'.log'))  
   testLogger = optim.Logger(paths.concat(opt.save, 'test'..opt.batchSize..'.log'))
   ModelUpdateLogger = optim.Logger(paths.concat(opt.save, 'ModelUpdateLog'..opt.batchSize..'.log'))
   ModelUpdateLogger:setNames{'iteration saved', 'validation error'}
end


-- This modifies model and logger
function train_validate_max_epochs(opt, trainData, validateData, model, criterion,
				   output_filename, train_logger, val_logger, model_update_logger)
   accuracy_tracker = {}
   old_accuracy = 0.0
   
   print '==> defining some tools'
   optimMethod, optimState = choose_optim_method(opt)
   
   for epoch = 1,MAX_EPOCHS do
      print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
      -- train.lua
      train_one_epoch(opt, trainData, optimMethod, optimState, model, criterion, train_logger)
      -- test.lua
      val_confusion = evaluate_model(opt, validateData, model, val_logger)
      
      new_accuracy = val_confusion.totalValid 
      table.insert(accuracy_tracker, new_accuracy)

      -- save model if accuracy is high enough
      savemodel(model, output_filename, new_accuracy, old_accuracy, epoch, model_update_logger)
      old_accuracy = new_accuracy
   end
--TODO   print(accuracy_tracker) 
end


function change_batch_size()
   -- prepare_data.lua
   local trainData, validateData, testData = build_datasets(
      opt.size, opt.tr_frac, opt.raw_train_data, opt.raw_test_data)
   -- prepare_model.lua
   local model = build_model(opt.model, trainData.mean, trainData.std)
   local criterion = build_criterion(opt.loss, trainData, validateData, testData, model)

   -- experiment with difference batch sizes
   for i = 1, #opt.batchSizeArray do
      -- set specific batchsize for expirement
      opt.batchSize = opt.batchSizeArray[i]
      local output_filename = paths.concat(opt.save, 'model'..opt.batchSize..'.net')
      -- change save path to folder for specific batchsize
      start_logging()
      -- train and run validation
      train_validate_max_epochs(opt, trainData, validateData, model, criterion,
				output_filename, trainLogger, valLogger, ModelUpdateLogger)
      -- see how the model does on the test data
      local test_confusion = evaluate_model(opt, testData, model, testLogger)
      print('\n\n')
      print('Test data performance')
      print(test_confusion)
   end
end

change_batch_size()
