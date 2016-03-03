
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



-------- added augmentation functions

function stack_tensors(tensor_A , tensor_B) -- first dimension
    -- stacks two tensors along the first dimension

    -- create a new matrix 
    combined_tensor = torch.Tensor(tensor_A:size(1)+tensor_B:size(1),tensor_A:size(2),
    tensor_A:size(3),tensor_A:size(4))

    -- populate with both previous tensors
    combined_tensor[{{1,tensor_A:size(1)}}] = tensor_A

    combined_tensor[{{tensor_A:size(1)+1,combined_tensor:size(1)}}] = tensor_B

    return combined_tensor
end

function stack_labels(tensor_A,tensor_B)
    combined_tensor = torch.Tensor(tensor_A:size(1)+tensor_B:size(1))
    combined_tensor[{{1,tensor_A:size(1)}}] = tensor_A
    combined_tensor[{{tensor_A:size(1)+1,combined_tensor:size(1)}}] = tensor_B
    return combined_tensor
end 




function do_something_redux(src_image)
    -- randomly transforms a single image

    local new = src_image:clone()

    -- rotate
    if torch.uniform() > 0.5 then
        angle_rad = 0.35*torch.uniform()
        --calculate new crop margin
        width = (src_image:size(2))/(math.cos(angle_rad)+math.sin(angle_rad))
        crop_margin = (new:size(2)-width)/2
        --choose roation direction
        if  torch.uniform() > 0.5 then
           new = image.rotate(new, angle_rad) -- up to a 20 degree angle    
        else
           new = image.rotate(new, -angle_rad) -- up to a 20 degree angle   
    end
        new = image.crop(new,crop_margin,crop_margin,new:size(2)-crop_margin,new:size(3)-crop_margin)
        new = image.scale(new, src_image:size(2), src_image:size(3))
    end
    
    -- change the hue
    
    --[[if torch.uniform() > 0.5 then
        new = image.rgb2hsv(new)
        if torch.uniform() > 0.5 then
            new[1] = new[1]-0.1
        else
            new[1] = new[1]+0.1
        end
        new = image.hsv2rgb(new)
    end]]
   
 
    -- tanslate
    
    if torch.uniform() > 0.5 then
        translate_by  = 12
        new = image.translate(new, translate_by, 0)
        new = image.crop(new,translate_by,0,new:size(2),new:size(3))
        new = image.scale(new, src_image:size(2), src_image:size(3))

    end
    

    -- crop

    if torch.uniform() > 0.5 then
        local cropping_pix = 12
        local cropped = image.crop(new,cropping_pix,cropping_pix,new:size(2)-cropping_pix,new:size(3)-cropping_pix)
        new = image.scale(cropped, src_image:size(2), src_image:size(3))
        -- local mlp = nn.Sequential()
        --mlp:add(nn.Padding(2,cropping_pix))
        --mlp:add(nn.Padding(3,cropping_pix))
        --mlp:add(nn.Padding(2,-cropping_pix))
        --mlp:add(nn.Padding(3,-cropping_pix))
        --new = mlp:forward(cropped)
    end
    return new
end


function augmented_all(data_input, label_input)
    local new_data= data_input:clone()

    -- transforms all 
    for i=1,new_data:size(1) do
        new_data[i] = do_something_redux(new_data[i])
    end
    local stacked_data = stack_tensors(data_input , new_data)
    local stacked_labels = stack_labels(label_input,label_input)
    return stacked_data, stacked_labels
end





