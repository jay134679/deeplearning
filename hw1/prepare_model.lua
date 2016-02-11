-- Homework 1: prepare_model.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)


require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

-- model_type must be 'linear', 'mlp', or 'convnet'.
-- 'normalized_data_mean' and 'normalized_data_std' are saved as
-- values in the model so they will be saved with the model when it is
-- written to disk. These can then be used to normalize test data when
-- the model is loaded from a file.
function build_model(model_type, normalized_data_mean, normalized_data_std)

   print '==> define parameters'

   -- 10-class problem
   local noutputs = 10
   
   -- input dimensions
   local nfeats = 1
   local width = 32
   local height = 32
   local ninputs = nfeats * width * height

   ----------------------------------------------------------------------
   print '==> construct model'

   local model = nn.Sequential()
   
   if model_type == 'linear' then
      -- Simple linear model
      model:add(nn.Reshape(ninputs))
      model:add(nn.Linear(ninputs,noutputs))
   elseif model_type == 'mlp' then
      -- number of hidden units (for MLP only):
      nhiddens = ninputs / 2

      -- Simple 2-layer neural network, with tanh hidden units
      model:add(nn.Reshape(ninputs))
      model:add(nn.Linear(ninputs,nhiddens))
      model:add(nn.Tanh())
      model:add(nn.Linear(nhiddens,noutputs))
   elseif model_type == 'convnet' then
      -- a typical convolutional network, with locally-normalized hidden
      -- units, and L2-pooling

      -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
      -- work on the SVHN dataset (http://arxiv.org/abs/1204.3968). In particular
      -- the use of LP-pooling (with P=2) has a very positive impact on
      -- generalization. Normalization is not done exactly as proposed in
      -- the paper, and low-level (first layer) features are not fed to
      -- the classifier.

      -- hidden units, filter sizes (for ConvNet only):
      local nstates = {64,64,128}
      local filtsize = 5
      local poolsize = 2
      local normkernel = image.gaussian1D(7)
      
      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
      model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
      model:add(nn.Tanh())
      model:add(nn.Linear(nstates[3], noutputs))



   elseif model_type == 'convnet_wdropout' then
      -- a typical convolutional network, with locally-normalized hidden
      -- units, and L2-pooling

      -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
      -- work on the SVHN dataset (http://arxiv.org/abs/1204.3968). In particular
      -- the use of LP-pooling (with P=2) has a very positive impact on
      -- generalization. Normalization is not done exactly as proposed in
      -- the paper, and low-level (first layer) features are not fed to
      -- the classifier.

      -- hidden units, filter sizes (for ConvNet only):
      local nstates = {64,64,128}
      local filtsize = 5
      local poolsize = 2
      local normkernel = image.gaussian1D(7)

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
      model:add(nn.Tanh())
      model:add(nn.Linear(nstates[3], noutputs))


   elseif model_type == 'convnet_filterchange' then
      -- a typical convolutional network, with locally-normalized hidden
      -- units, and L2-pooling

      -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
      -- work on the SVHN dataset (http://arxiv.org/abs/1204.3968). In particular
      -- the use of LP-pooling (with P=2) has a very positive impact on
      -- generalization. Normalization is not done exactly as proposed in
      -- the paper, and low-level (first layer) features are not fed to
      -- the classifier.

      -- hidden units, filter sizes (for ConvNet only):
      local nstates = {128,128,128}
      local filtsize = 5
      local poolsize = 2
      local normkernel = image.gaussian1D(7)

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(nstates[2]*filtsize*filtsize)) 
      --model:add(nn.Dropout(0.5))
      model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
      model:add(nn.Tanh())
      model:add(nn.Linear(nstates[3], noutputs))

   elseif model_type == 'convnet_long' then
      -- a typical convolutional network, with locally-normalized hidden
      -- units, and L2-pooling

      -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
      -- work on the SVHN dataset (http://arxiv.org/abs/1204.3968). In particular
      -- the use of LP-pooling (with P=2) has a very positive impact on
      -- generalization. Normalization is not done exactly as proposed in
      -- the paper, and low-level (first layer) features are not fed to
      -- the classifier.

      -- hidden units, filter sizes (for ConvNet only):
      local nstates = {128,128,128}
      local filtsize = 5
      local poolsize = 2
      local normkernel = image.gaussian1D(7)

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
      model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
      model:add(nn.Tanh())
      model:add(nn.Linear(nstates[3], noutputs))


   else
      error('unknown -model')
   end

   -- Saving the normalized mean and standard deviation to prepare testing data.
   model['normalized_data_mean'] =  normalized_data_mean
   model['normalized_data_std'] = normalized_data_std
   
   ----------------------------------------------------------------------
   print '==> here is the model:'
   print(model)
   return model
end


function build_criterion(loss, trainData, validateData, testData, model)
   -- 10-class problem
   local noutputs = 10

   ----------------------------------------------------------------------
   print '==> define loss'

   local criterion = nil
   
   if loss == 'margin' then

      -- This loss takes a vector of classes, and the index of
      -- the grountruth class as arguments. It is an SVM-like loss
      -- with a default margin of 1.

      criterion = nn.MultiMarginCriterion()

   elseif loss == 'nll' then

      -- This loss requires the outputs of the trainable model to
      -- be properly normalized log-probabilities, which can be
      -- achieved using a softmax function

      model:add(nn.LogSoftMax())

      -- The loss works like the MultiMarginCriterion: it takes
      -- a vector of classes, and the index of the grountruth class
      -- as arguments.

      criterion = nn.ClassNLLCriterion()

   elseif loss == 'mse' then

      -- for MSE, we add a tanh, to restrict the model's output
      model:add(nn.Tanh())

      -- The mean-square error is not recommended for classification
      -- tasks, as it typically tries to do too much, by exactly modeling
      -- the 1-of-N distribution. For the sake of showing more examples,
      -- we still provide it here:

      criterion = nn.MSECriterion()
      criterion.sizeAverage = false

      -- Compared to the other losses, the MSE criterion needs a distribution
      -- as a target, instead of an index. Indeed, it is a regression loss!
      -- So we need to transform the entire label vectors:

      if trainData then
	 -- convert training labels:
	 local trsize = (#trainData.labels)[1]
	 local trlabels = torch.Tensor( trsize, noutputs )
	 trlabels:fill(-1)
	 for i = 1,trsize do
	    trlabels[{ i,trainData.labels[i] }] = 1
	 end
	 trainData.labels = trlabels

	 -- convert valdiation labels
	 local valsize = (#validateData.labels)[1]
	 local vallabels = torch.Tensor( valsize, noutputs )
	 vallabels:fill(-1)
	 for i = 1,valsize do
	    vallabels[{ i,ValData.labels[i] }] = 1
	 end
	 validateData.labels = vallabels

	 -- convert test labels
	 local tesize = (#testData.labels)[1]
	 local telabels = torch.Tensor( tesize, noutputs )
	 telabels:fill(-1)
	 for i = 1,tesize do
	    telabels[{ i,testData.labels[i] }] = 1
	 end
	 testData.labels = telabels
      end

   else

      error('unknown -loss')

   end

   ----------------------------------------------------------------------
   print '==> here is the loss function:'
   print(criterion)
   return criterion
end
