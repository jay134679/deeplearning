-- Homework 4: nngraph_warmup.lua
-- Deep Learning Spring 2016
-- Alex Pine (akp258@nyu.edu)

require 'nngraph'
require 'torch'

X_SIZE = 4
Y_SIZE = 5
A_SIZE = 2

-- Returns a model that expects three input vectors. The first should be of
-- length X_SIZE, the second Y_SIZE, and the third A_SIZE.
function problem1a()
   local in1 = nn.Linear(X_SIZE, A_SIZE)()
   in1.data.module.weight = torch.ones(A_SIZE, X_SIZE)
   in1.data.module.bias = torch.ones(A_SIZE)
   local in2 = nn.Linear(Y_SIZE, A_SIZE)()
   in2.data.module.weight = torch.ones(A_SIZE, Y_SIZE)
   in2.data.module.bias = torch.ones(A_SIZE)
   local in3 = nn.Identity()()
   local left = nn.Square()(nn.Tanh()(in1))
   local right = nn.Square()(nn.Sigmoid()(in2))
   local mul_result = nn.CMulTable()({left, right})
   local out = nn.CAddTable()({mul_result, in3})

   return nn.gModule({in1, in2, in3}, {out})
end

-- Takes three vectors of length X_SIZE, Y_SIZE, and A_SIZE respectively, builds
-- a model using problem1a(), prints the foward output, and prints the backward
-- output with gradient input of ones.
function problem1b(x, y, z)
   print('Input x:')
   print(x)
   print('Input y:')
   print(y)
   print('Input z:')
   print(z)
   local model = problem1a()
   local forwardOutput = model:forward({x, y, z})
   print('Forward output:')
   print(forwardOutput)
   local gradInput = torch.DoubleTensor(2):fill(1)
   local backOutput = model:backward({x, y, z}, gradInput)
   print('Backwards output:')
   for k, v in ipairs(backOutput) do
      print('Back output '..k..':')
      print(v)
   end
end

xIn = torch.ones(X_SIZE)
yIn = torch.ones(Y_SIZE)
zIn = torch.ones(A_SIZE)
problem1b(xIn, yIn, zIn)
