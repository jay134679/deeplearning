-- Homework 2: main.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
-- Trains, validates, and tests data for homework 2.

--require 'lapp'
require 'torch'

function parse_cmdline()
   local opt = lapp[[
      --results_dir             (default "results")   directory to save results
      --debug_log_filename      (default "debug.log")  filename of debugging output
      --exp_name                (default "")          name of the current experiment. optional.
      -b,--batchSize            (default 64)          batch size
      -r,--learningRate         (default 1)        learning rate
      --learningRateDecay       (default 1e-7)      learning rate decay
      --weightDecay             (default 0.0005)      weightDecay
      -m,--momentum             (default 0.9)         momentum
      --epoch_step              (default 25)          epoch step
      --model                  (default vgg_bn_drop)     model name
      --max_epoc                (default 300)           maximum number of iterations
      --backend                 (default nn)            backend
   ]]
   return opt
end

-- creates the 'results_dir' directory if it doesn't exist, and creates the current
-- experiment's directory, and returns its name.
function prepare_environment(opt)
   paths.mkdir(opt.results_dir)  -- TODO what happens if the dir already exists

   local full_exp_name = ''
   if opt.exp_name ~= '' then
      full_exp_name = opt.exp_name
   else
      full_exp_name = 'exp'
   end
   full_exp_name = full_exp_name..'_'..os.date("%m%d%H%M%S")
   local experiment_dir = paths.concat(opt.results_dir, full_exp_name)
   print('Saving results at '..experiment_dir)
   paths.mkdir(experiment_dir)
   return experiment_dir
end

-- Creates the debug log file, and sets it to be the default output for
-- io.write() calls.
function create_debug_log(debug_log_filename, experiment_dir)
   -- set file for io.write calls. 'a' is for 'append' mode.
   debug_file = io.open(paths.concat(experiment_dir, debug_log_filename), 'a')
   io.output(debug_file)
   return debug_file
end

-- Use io.write('blah') to write debug output
-- save files in experiment_dir
function main()
   opt = parse_cmdline()
   experiment_dir = prepare_environment(opt)
   debug_file = create_debug_log(opt.debug_log_filename, experiment_dir)
   io.close(debug_file)
end

main()
