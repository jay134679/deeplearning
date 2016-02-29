-- Homework 2: exp_setup.lua
-- Maya Rotmensch (mer567) and Alex Pine (akp258)
-- Creates functions for setting up the experimental environment.

-- Usage:
-- call setup_experiment, then call DEBUG('blah') to write debug output.

------ private functions -----

-- creates the 'results_dir' directory if it doesn't exist, and creates the current
-- experiment's directory, and returns its name.
local function prepare_experiment_dir(results_dir, exp_name)
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
-- DEBUG(message) can be called once this has been called
local function CREATE_DEBUG_LOG(debug_log_filename, experiment_dir)
   -- set file for io.write calls. 'a' is for 'append' mode.
   local filepath = paths.concat(experiment_dir, debug_log_filename)
   -- global variable
   DEBUG_FILE = assert(io.open(filepath, 'a'))
end

-- saves input options to options.log
local function save_input_options(opt, experiment_dir)
   local options_str = 'Options\n'
   for k,v in pairs(opt) do
      options_str = options_str..k..': '..tostring(v)..'\n'
   end
   local options_filename = paths.concat(experiment_dir, 'options.log')
   DEBUG('Writing options to '..options_filename)
   local options_file = assert(io.open(options_filename, 'a'))
   options_file:write(options_str)
   options_file:close()
end

----- public functions ------

-- creates experiment directory, and sets up the global DEBUG function.
-- Returns experiment_dir.
function setup_experiment(opt)
   experiment_dir = prepare_experiment_dir(opt.results_dir, opt.exp_name)
   CREATE_DEBUG_LOG(opt.debug_log_filename, experiment_dir)
   save_input_options(opt, experiment_dir)
   return experiment_dir
end

-- Prints the given message, and writes it to the debug file if it's been
-- defined by CREATE_DEBUG_LOG().
function DEBUG(message)
   print(message)
   if DEBUG_FILE ~= nil then
      DEBUG_FILE:write(tostring(message)..'\n')
   end
end
