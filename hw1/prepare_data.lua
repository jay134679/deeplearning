-- Homework 1: prepare_data.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)

require 'torch'

-- THis function  tries to load data from a directory 'mnist.t7', and downloads
-- it if it's not there. Once the data files have been acquired, they are loaded
-- into tensors. The training set is split into training data into a smaller
-- training set and a validation set based on 'size' and 'tr_frac'. 'size'
-- determines how much of the data files are loaded, and 'tr_frac' is a float
-- between 0 and 1 that determines how much of the training data is actually
-- used for training. The remainder, 1-tr_frac, is used as a validation set.
-- Finally, this function returns three tensors: the training data, the
-- validation data, and the test data.
function build_datasets(size, tr_frac)
   local data_path = 'mnist.t7'
   local train_file = paths.concat(data_path, 'train_32x32.t7')
   local test_file = paths.concat(data_path, 'test_32x32.t7')
   
   if not paths.filep(train_file) or not paths.filep(test_file) then
      print '==> downloading dataset'
      local tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
      os.execute('wget ' .. tar)
      os.execute('tar xvf ' .. paths.basename(tar))
   end
   local raw_train_data = torch.load(train_file, 'ascii')
   local raw_test_data = torch.load(test_file, 'ascii')
   
   -- figure out how much of the total data to use according to 'size'.
   local trsize = 0
   local tesize = 0
   if size == 'full' then
      print '==> using regular, full training data'
      trsize = raw_train_data.data:size(1)
      tesize = raw_test_data.data:size(1)
      -- the size of these files is known ahead of time, double check.
      assert(trsize == 60000)
      assert(tesize == 10000)
   elseif size == 'small' then
      print '==> using reduced training data, for fast experiments'
      trsize = 6000
      tesize = 1000
   elseif size == 'tiny' then
      print '==> using tiny training data, for initial experiments'
      trsize = 600
      tesize = 100
   end
   if trsize == 0 then
      error("ERROR: unregconized value for 'size' string. Choose 'full', 'small', or 'tiny'.")
   end

   -- Now that we have the desired data sizes, resize the train and validation
   -- data by the tr_frac.
   local new_trsize = tr_frac * trsize
   local new_valsize = trsize - new_trsize

   -- split training into train and validate
   local full_train_size = tr_frac * raw_train_data.data:size(1)
   local split_train_data_table = torch.split(raw_train_data.data, full_train_size, 1)
   local split_train_label_table = raw_train_data.labels:split(full_train_size, 1)
   
   local trainData = {
      data = split_train_data_table[1]:float(),
      labels = split_train_label_table[1],
      size = function() return new_trsize end
   }
   local validateData = {
      data = split_train_data_table[2]:float(),
      labels = split_train_label_table[2],
      size = function() return new_valsize end
   }

   local testData = {
      data = raw_test_data.data:float(),
      labels = raw_test_data.labels,
      size = function() return tesize end
   }

   -- Normalize each channel, and store mean/std.
   -- These values are important, as they are part of
   -- the trainable parameters. At test time, test data will be normalized
   -- using these values.
   print '==> preprocessing data: normalize globally'
   local mean = trainData.data[{ {},1,{},{} }]:mean()
   local std = trainData.data[{ {},1,{},{} }]:std()
   
   trainData.data[{ {},1,{},{} }]:add(-mean)
   trainData.data[{ {},1,{},{} }]:div(std)
   
   -- Normalize test data, using the training means/stds
   testData.data[{ {},1,{},{} }]:add(-mean)
   testData.data[{ {},1,{},{} }]:div(std)
   
   -- normalize validation data
   validateData.data[{ {},1,{},{} }]:add(-mean)
   validateData.data[{ {},1,{},{} }]:div(std)
      
   ----------------------------------------------------------------------
   print '==> verify statistics'
   
   trainData.mean = trainData.data[{ {},1 }]:mean()
   trainData.std = trainData.data[{ {},1 }]:std()

   validateData.mean =  validateData.data[{ {},1 }]:mean()
   validateData.std = validateData.data[{ {},1 }]:std()
   
   testData.mean = testData.data[{ {},1 }]:mean()
   testData.std = testData.data[{ {},1 }]:std()
      
   -- It's always good practice to verify that data is properly
   -- normalized.   
   print('training data mean: ' .. trainData.mean)
   print('training data standard deviation: ' .. trainData.std)
   assert(math.abs(trainData.mean) < 0.1)
   assert(math.abs(trainData.std - 1.0) < 0.1)

   print('validation data mean: ' .. validateData.mean)
   print('validation data standard deviation: ' .. validateData.std)
   assert(math.abs(validateData.mean) < 0.1)
   assert(math.abs(validateData.std - 1.0) < 0.1)
   
   print('test data mean: ' .. testData.mean)
   print('test data standard deviation: ' .. testData.std)
   assert(math.abs(testData.mean) < 0.1)
   assert(math.abs(testData.std - 1.0) < 0.1)
   
   return trainData, validateData, testData
end
