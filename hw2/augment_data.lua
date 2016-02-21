
require 'nn'
require 'torch'

-- Create the BatchFlip model layer, and add it to the nn module.
do
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end  
end

-- Adds the BatchFlip data augmentation layer to the given model.
function add_batch_flip(model)
   model:add(nn.BatchFlip():float())
end

