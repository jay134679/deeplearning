-- Homework 1: result.lua
-- Loads a trained MNIST model, and makes predictions on the test data in
-- 'mnist.t7/test_32x32.t7'.


-- TODO load model
model = torch.load(opt.model) -- TODO opt
-- TODO load test data
test_file = 'mnist.t7/test_32x32.t7'
-- TODO load test data
-- TODO load train data
-- TODO normalize the training data with the test data? seems weird.

-- make predictions
for t = 1, testData:size() do
   -- get new sample
   local input = testData.data[t]
   if opt.type == 'double' then  -- TODO opt
      input = input:double()
   elseif opt.type == 'cuda' then
      input = input:cuda()
   end
   local prediction = model:forward(input)
   -- TODO put this in a list
end
   
   
-- TODO print output to 'predictions.csv' (this gets uploaded to kaggle)
-- e.g.
-- Id,Prediction
-- 1,5
-- 2,2
-- 3,1
-- 4,10
-- 5,6
