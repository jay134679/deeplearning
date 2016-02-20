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
function prepare_environment(results_dir, exp_name)
   paths.mkdir(results_dir)
   local full_exp_name = ''
   if exp_name ~= '' then
      full_exp_name = exp_name
   else
      full_exp_name = 'exp'
   end
   full_exp_name = full_exp_name..'_'..os.date("%m%d%H%M%S")
   local experiment_dir = paths.concat(results_dir, full_exp_name)
   print('Saving results at '..experiment_dir)
   paths.mkdir(experiment_dir)
   return experiment_dir
end

-- Creates the global debug log file, DEBUG_FILE.
-- LOG(message) can be called once this has been called
function CREATE_DEBUG_LOG(debug_log_filename, experiment_dir)
   -- set file for io.write calls. 'a' is for 'append' mode.
   local filepath = paths.concat(experiment_dir, debug_log_filename)
   -- global variable
   DEBUG_FILE = assert(io.open(filepath, 'a'))
end

-- Uses io.write to log the given message on a single line.
-- Assumes CREATE_DEBUG_LOG has already been called.
function LOG(message)
   if DEBUG_FILE ~= nil then
      DEBUG_FILE:write(message..'\n')
   else
      print('[ERROR] DEBUG_FILE global var is nil')
   end
end

-- saves input options to options.log
function save_input_options(opt, experiment_dir)
   local options_str = 'Options\n'
   for k,v in pairs(opt) do
      options_str = options_str..k..': '..v..'\n'
   end
   local options_filename = paths.concat(experiment_dir, 'options.log')
   LOG('Writing options to '..options_filename)
   local options_file = assert(io.open(options_filename, 'a'))
   options_file:write(options_str)
   options_file:close()
end

-- Use LOG('blah') to write debug output
-- save files in experiment_dir
function main()
   opt = parse_cmdline()
   experiment_dir = prepare_environment(opt.results_dir, opt.exp_name)
   CREATE_DEBUG_LOG(opt.debug_log_filename, experiment_dir)
   save_input_options(opt, experiment_dir)
   print('Experiment complete.')
end

main()
