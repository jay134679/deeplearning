-- Homework 2: provider.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
-- Original code written by jakezhao.
-- Data loading and parsing code.

-- TODO allow data to be loaded separately. All full data OOMs on AWS.

require 'nn'
require 'image'
require 'torch'
require 'xlua'

require 'exp_setup'

torch.setdefaulttensortype('torch.FloatTensor')

-- Public function to load the Provider object. If its been saved to a file
-- already, it uses that.
function load_provider(size, providerType)
   DEBUG('==> loading data')
   local data_filename = 'provider.'..size..'.'..providerType..'.t7'
   local data_file = io.open(data_filename, 'r')
   local provider = nil
   if data_file ~= nil then
      DEBUG('loading data from file...')
      provider = torch.load(data_filename)
   else
      DEBUG('downloading data...')
      provider = Provider(size, providerType)
      provider:normalize()
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
function Provider:__init(size, providerType)
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
   
   if (providerType == 'training') then
     raw_val = torch.load('stl-10/val.t7b')
     raw_extra = torch.load('stl-10/extra.t7b')
   elseif (providerType == 'evaluate') then
     raw_test = torch.load('stl-10/test.t7b')
   else
     error("[ERROR] unregconized value for 'providerType': "..providerType..". Choose 'training' or 'evaluate'")
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
      extrasize = 100000
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
      extrasize = 2000
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
      size = function() return trsize end
   }
   self.trainData.data, self.trainData.labels = parseDataLabel(
      raw_train.data, trsize, channel, height, width)
   self.trainData.data = self.trainData.data:float()
   self.trainData.labels = self.trainData.labels:float()
   
   -- validation

   if (providerType == 'training') then
     DEBUG('loading validation data')
     self.valData = {
        data = torch.Tensor(),
        labels = torch.Tensor(),
        size = function() return valsize end
     }
     self.valData.data, self.valData.labels = parseDataLabel(
        raw_val.data, valsize, channel, height, width)	

     -- extra
     DEBUG('unlabeled data')
     self.extraData = {
        data = torch.Tensor(),
        size = function() return extrasize end
     }
     self.extraData.data = parseUnlabeledData(
        raw_extra.data, extrasize, channel, height, width)

     -- convert from ByteTensor to Float
     self.valData.data = self.valData.data:float()
     self.valData.labels = self.valData.labels:float()
     self.extraData.data = self.extraData.data:float()
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

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/val sets
  --
  DEBUG('<trainer> preprocessing data (color space + normalization)')
  collectgarbage()
  local trainData = self.trainData
  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v

  -- preprocess either validate or test, depending on the providerType
  local providerType = self.providerType

  if providerType == 'training' then
    local valData = self.valData

    -- preprocess valSet
    for i = 1,valData:size() do
      xlua.progress(i, valData:size())
       -- rgb -> yuv
       local rgb = valData.data[i]
       local yuv = image.rgb2yuv(rgb)
       -- normalize y locally:
       yuv[{1}] = normalization(yuv[{{1}}])
       valData.data[i] = yuv
    end
    -- normalize u globally:
    valData.data:select(2,2):add(-mean_u)
    valData.data:select(2,2):div(std_u)
    -- normalize v globally:
    valData.data:select(2,3):add(-mean_v)
    valData.data:select(2,3):div(std_v)

    local extraData = self.extraData
    -- TODO normalize unlabeled data with the training data?

  elseif providerType == 'evaluate' then
    local testData = self.testData

    -- preprocess testSet
    for i = 1,testData:size() do
      xlua.progress(i, testData:size())
       -- rgb -> yuv
       local rgb = testData.data[i]
       local yuv = image.rgb2yuv(rgb)
       -- normalize y locally:
       yuv[{1}] = normalization(yuv[{{1}}])
       testData.data[i] = yuv
    end
    -- normalize u globally:
    testData.data:select(2,2):add(-mean_u)
    testData.data:select(2,2):div(std_u)
    -- normalize v globally:
    testData.data:select(2,3):add(-mean_v)
    testData.data:select(2,3):div(std_v)
  end
end
