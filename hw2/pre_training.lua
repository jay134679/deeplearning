
-- 
require 'torch'
require 'image'
require 'unsup'
require 'provider'
require 'exp_setup'

function parse_cmdline()
   local opt = lapp[[
      --size                  (default "tiny")      size of data to use: tiny, small, full.
      --exp_name              (default "pre_training")          name of the current experiment. optional.
      --no_cuda                                     whether to use cuda or not. defaults to false, so cuda is used by default.
      --results_dir           (default "results")   directory to save results
      --kmeans_threshold      (default 140)         a scalar  determining the min mean approximate gradient of the patch(want only edges)
      --debug_log_filename    (default "pre_training_debug.log")  filename of debugging output
      -b, --batch_size        (default 20)          batch size
      -k, --kernels           (default 64)          number of "kernels" (centroids) you want from kmeans
      -n, --niter             (default 1000)        number of kmeans iterations
      -a --augmented                                defaults to false
   ]]
   return opt
end


function pad_sobel(sobelfied)
    --[[adds a pixel of black padding 
    (to avoid the while padding we get from applying the sobel filter for size "full")]]
    local mlp = nn.Sequential()
    mlp:add(nn.Padding(2,1))
    mlp:add(nn.Padding(3,1))
    mlp:add(nn.Padding(2,-1))
    mlp:add(nn.Padding(3,-1))
    new = mlp:forward(sobelfied)
    return new
end

-- apply a sobel filter on a single image
function sobel_operator(src_image)
    -- function that applies a sobel filter to an image
    local Gx = torch.Tensor({{-1,0,1},{-2,0,2},{-1,0,1}}) --horizontal
    local Gy = torch.Tensor({{-1,-2,-1},{0,0,0},{1,2,1}}) --vertical
    local conv_Gx = image.convolve(src_image, Gx, 'valid')
    local conv_Gy = image.convolve(src_image, Gy, 'valid')
    local sobelfied = torch.sqrt(torch.pow(conv_Gx,2)+torch.pow(conv_Gy,2))
    return pad_sobel(sobelfied)
end




function extract_patches_batch(src_images_tensor, threshold)
    --[[ sobelfied_image input is a table of images who have been run throguh the sobel filter.
    threshold is a scalar  determining the mean approximate gradient of the patch]]
    
    local stride = 3
    local patch_size = 5
    local chosen_patches = {}
    
    local start_height_pixel= 1
    local start_width_pixel = 1
    local sobelfied_image = nil
    local src_image = nil
    
    for image_ct =1, src_images_tensor:size(1) do
 
        src_image = src_images_tensor[image_ct] -- get image
    
        sobelfied_image = sobel_operator(src_image) -- apply sobel filter
        
        start_height_pixel= 1
        start_width_pixel = 1
        

        while start_height_pixel+patch_size<= sobelfied_image:size(2) do 
            while start_width_pixel+patch_size<=  sobelfied_image:size(3) do 

                local patch_considered = sobelfied_image[{{},{start_height_pixel,start_height_pixel+patch_size -1},
                            {start_width_pixel,start_width_pixel+patch_size -1}}]

                mean_grad = torch.mean(patch_considered)
                
                if mean_grad > threshold then --check if this is an edge
                    
                    -- retrieved corresponding patch from original image
                    local original_patch = src_image[{{},{start_height_pixel,start_height_pixel+patch_size -1},
                            {start_width_pixel,start_width_pixel+patch_size -1}}]
                    
                    table.insert(chosen_patches,original_patch)
                end

                start_width_pixel = start_width_pixel+stride -- advance pointer
            end
            
            start_width_pixel = 1
            start_height_pixel = start_height_pixel+stride
        end

        if image_ct%100==0 then
            print(image_ct)
            collectgarbage()
        end
    end

    collectgarbage()
    print("done")
    
    -- stack valid patches
    local chosen_patches_tensor = torch.Tensor(#chosen_patches, patch_size*patch_size*3)
    for i=1 , #chosen_patches do
        local a = torch.reshape(chosen_patches[i], 1, chosen_patches_tensor:size(2))
        chosen_patches_tensor[{{i},{}}] = a
    end
    return chosen_patches_tensor
end



function main()
   opt = parse_cmdline()
   experiment_dir = setup_experiment(opt)
   -- DEBUG function now callable
   
   --save_input_options(opt, experiment_dir)
   provider = load_provider(opt.size, 'unlabeled',opt.augmented)
   -- run throguh sobel filter
   print(provider.extraData.data:size())
   --chosen_patches_tensor = extract_patches_batch(provider.extraData.data[{{1,10},{},{},{}}], opt.kmeans_threshold) 
   chosen_patches_tensor = extract_patches_batch(provider.extraData.data, opt.kmeans_threshold)
   print (chosen_patches_tensor:size())
   --print(opt.kernels, opt.niter, opt.batch_size)

   centroids,totalcounts = unsup.kmeans(chosen_patches_tensor, opt.kernels, opt.niter, opt.batch_size, false, true)
   print(totalcounts)
   torch.save('kmeans.t7',centroids)
   print('Experiment complete.')
end

main()

--chosen_patches_tensor = extract_patches_batch(temp, 70)
