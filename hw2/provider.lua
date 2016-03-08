-- Homework 2: provider.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
-- Original code written by jakezhao.
-- Data loading and parsing code.

require 'nn'
require 'image'
require 'torch'
require 'xlua'
require 'augment_data'
local c = require 'trepl.colorize'

require 'exp_setup'

torch.setdefaulttensortype('torch.FloatTensor')

-- Public function to load the Provider object. If its been saved to a file
-- already, it uses that.

-- providerTypes:
-- training: trainData, valData
-- unlabeled: trainData, extraData, valData
-- evaluate, trainData, testData
function load_provider(size, providerType, augmented)
   DEBUG('==> loading data')
   local data_filename = ''
   if providerType == 'training' or providerType == 'unlabeled' then
      if augmented then
         data_filename = 'provider.'..size..'.'..providerType..'.augmented.t7'
	 DEBUG('loading augmented data file')
      else
         DEBUG("training not augmented")
         data_filename = 'provider.'..size..'.'..providerType..'.t7'
      end
   else
      data_filename = 'provider.'..size..'.'..providerType..'.t7'
   end
   DEBUG('Loading data from file: '..data_filename) 
   local data_file = io.open(data_filename, 'r')
   local provider = nil
   if data_file ~= nil then
      DEBUG('loading data from file...')
      provider = torch.load(data_filename)
   else
      DEBUG('downloading data...')
      provider = Provider(size, providerType, augmented)
      if augmented then
         DEBUG('Augmenting labeled data...')
         provider.trainData.data, provider.trainData.labels = augmented_all(provider.trainData.data,
									    provider.trainData.labels)
	 if providerType == 'unlabeled' then
	    DEBUG('Augmenting unlabeled data...')
	    provider.extraData.data = augmented_all(provider.extraData.data)
	 end
      end
      provider:normalize() -- TO DO 
      torch.save(data_filename, provider)
   end
   return provider
end

---- Private functions ----

-- parse STL-10 data from table into examples and labels tensors
function parseDataLabel(data, numSamples, numChannels, height, width)
   DEBUG('num samples '..numSamples)
   local examples = torch.ByteTensor(numSamples, numChannels, height, width)
   local labels = torch.ByteTensor(numSamples)
   local idx = 1
   done = false
   for i = 1, #data do
      local this_d = data[i]
      for j = 1, #this_d do
	 if idx > numSamples then
	    done = true
	    break
	 end
	 examples[idx]:copy(this_d[j])
	 labels[idx] = i
	 idx = idx + 1
      end
      if done then break end
   end
   return examples, labels
end

-- Parse unlabeled data into a byte tensor.
-- TODO seems weird that you have to do data[1][i].
function parseUnlabeledData(data, numSamples, numChannels, height, width)
   DEBUG('num samples '..numSamples)
   local examples = torch.ByteTensor(numSamples, numChannels, height, width)
   for i = 1, #data[1] do
      if i > numSamples then
	 break
      end
      examples[i]:copy(data[1][i])
   end
   return examples
end


local Provider = torch.class 'Provider'

-- size is either 'tiny', 'small', or 'full'.
-- provider must be 'training' or 'evaluate'.
-- 'training' loads train, val, and extra.
-- 'evaluate' loads train (for its mean and std) and test.
function Provider:__init(size, providerType, augmented)
  -- download dataset
   if not paths.dirp('stl-10') then
      os.execute('mkdir stl-10')
      local www = {
         train = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/train.t7b',
         val = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/val.t7b',
         extra = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/extra.t7b',
         test = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/test.t7b'
      }
      
      os.execute('wget ' .. www.train .. '; '.. 'mv train.t7b stl-10/train.t7b')
      os.execute('wget ' .. www.val .. '; '.. 'mv val.t7b stl-10/val.t7b')
      os.execute('wget ' .. www.test .. '; '.. 'mv test.t7b stl-10/test.t7b')
      os.execute('wget ' .. www.extra .. '; '.. 'mv extra.t7b stl-10/extra.t7b')
   end

   self.providerType = providerType
   -- Always load train
   local raw_train = torch.load('stl-10/train.t7b')
   -- Load other sets conditionally
   local raw_val = nil
   local raw_test = nil
   local raw_extra = nil
   
   if providerType == 'training' or providerType == 'unlabeled' then
     raw_val = torch.load('stl-10/val.t7b')
     if providerType == 'unlabeled' then
	raw_extra = torch.load('stl-10/extra.t7b')
     end
   elseif providerType == 'evaluate' then
     raw_test = torch.load('stl-10/test.t7b')
   else
     error("[ERROR] unregconized value for 'providerType': "..providerType..". Choose 'training', 'unlabeled' or 'evaluate'")
   end
   
   local trsize = 0
   local valsize = 0
   local testsize = 0
   local extrasize = 0
   
   if size == 'full' then
      DEBUG('==> using regular, full training data')
      trsize = 4000
      valsize = 1000
      testsize = 8000
      extrasize = 8000 -- TODO originally 100000, but it's too much!
   elseif size == 'small' then
      DEBUG('==> using reduced training data, for fast experiments')
      trsize = 1000
      valsize = 250
      testsize = 2000
      extrasize = 20000
   elseif size == 'tiny' then
      DEBUG('==> using tiny training data, for code testing')
      trsize = 100
      valsize = 25
      testsize = 200
      extrasize = 100
   end
   if trsize == 0 then
      error("[ERROR] unregconized value for 'size' string: "..size..". Choose 'full', 'small', or 'tiny'.")
   end

   local channel = 3
   local height = 96
   local width = 96
  
   -- load and parse dataset

   -- train
   DEBUG('loading training data')
   self.trainData = {
      data = torch.Tensor(),
      labels = torch.Tensor(),
      --size = function() return trsize end
      size = function () if augmented then return 2*trsize else return trsize end end
   }
   self.trainData.data, self.trainData.labels = parseDataLabel(
      raw_train.data, trsize, channel, height, width)
   self.trainData.data = self.trainData.data:float()
   self.trainData.labels = self.trainData.labels:float()
   
   if (providerType == 'training' or providerType == 'unlabeled') then
      -- validation
     DEBUG('loading validation data')
     self.valData = {
        data = torch.Tensor(),
        labels = torch.Tensor(),
        size = function() return valsize end
     }
     self.valData.data, self.valData.labels = parseDataLabel(
        raw_val.data, valsize, channel, height, width)	
     self.valData.data = self.valData.data:float()
     self.valData.labels = self.valData.labels:float()

     -- extra
     if providerType == 'unlabeled' then
	DEBUG('unlabeled data')
	self.extraData = {
	   data = torch.Tensor(),
	   size = function() return extrasize end
	}
	self.extraData.data = parseUnlabeledData(
	   raw_extra.data, extrasize, channel, height, width)
	-- convert from ByteTensor to Float
	self.extraData.data = self.extraData.data:float()
     end
   end
   
   -- test
   if (providerType == 'evaluate') then
     DEBUG('loading test data')
     self.testData = {
        data = torch.Tensor(),
        labels = torch.Tensor(),
        size = function() return testsize end
     }
     self.testData.data, self.testData.labels = parseDataLabel(
        raw_test.data, testsize, channel, height, width)
      -- convert from ByteTensor to Float
     self.testData.data = self.testData.data:float()
     self.testData.labels = self.testData.labels:float()
   end
   
   collectgarbage()
end


function transformRgbToYuv(dataObj, normalization)
   for i = 1, dataObj:size() do
      xlua.progress(i, dataObj:size())
      -- rgb -> yuv
      local rgb = dataObj.data[i]
      local yuv = image.rgb2yuv(rgb)
      -- normalize y locally:
      yuv[{1}] = normalization(yuv[{{1}}])
      dataObj.data[i] = yuv
   end
end   

function normalizeUVMeanAndStd(dataObj, mean_u, std_u, mean_v, std_v)
   -- normalize u globally:
   dataObj.data:select(2,2):add(-mean_u)
   dataObj.data:select(2,2):div(std_u)
   -- normalize v globally:
   dataObj.data:select(2,3):add(-mean_v)
   dataObj.data:select(2,3):div(std_v)
end


function Provider:normalize()
   ----------------------------------------------------------------------
   -- preprocess/normalize train/val sets
   --
   DEBUG('<trainer> preprocessing data (color space + normalization)')
   collectgarbage()
   
   local trainData = self.trainData
   DEBUG('Training data size: '..trainData:size())
   
   -- preprocess trainSet
   
   local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
   transformRgbToYuv(trainData, normalization)
   
   -- normalize u and v globally:
   local mean_u = trainData.data:select(2,2):mean()
   local std_u = trainData.data:select(2,2):std()
   local mean_v = trainData.data:select(2,3):mean()
   local std_v = trainData.data:select(2,3):std()
   normalizeUVMeanAndStd(trainData, mean_u, std_u, mean_v, std_v)
   -- saving the normalizing constants for evaluation later
   trainData.mean_u = mean_u
   trainData.std_u = std_u
   trainData.mean_v = mean_v
   trainData.std_v = std_v
   
   -- preprocess either validate or test, depending on the providerType
   if self.providerType == 'training' or self.providerType == 'unlabeled' then
      -- validation
      transformRgbToYuv(self.valData, normalization)
      normalizeUVMeanAndStd(self.valData, mean_u, std_u, mean_v, std_v)
      
      if self.providerType == 'unlabeled' then
	 -- unlabeled
	 transformRgbToYuv(self.extraData, normalization)
	 normalizeUVMeanAndStd(self.extraData, mean_u, std_u, mean_v, std_v)
      end
   elseif self.providerType == 'evaluate' then
      -- testing
      transformRgbToYuv(self.testData, normalization)
      normalizeUVMeanAndStd(self.testData, mean_u, std_u, mean_v, std_v)
   end
end

