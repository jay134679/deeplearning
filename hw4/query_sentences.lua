-- Homework 4:  query_sentences.lua
-- Deep Learning Spring 2016
-- Alex Pine (akp258@nyu.edu)

gpu = false

require 'io'
stringx = require('pl.stringx')
require 'nn'
require('nngraph')
require('base')
ptb = require('data')
require 'exp_setup'

function parse_cmdline()
   local opt = lapp[[
      --mode                  (default query)       either 'train' or 'query'. if 'query', specify the model_file.
      --exp_name              (default "")          name of the current experiment. optional.
      --model_file            (default "")          in test mode, use this file as the model. all other params are for 'train'.
      --model_save_freq       (default 50)          save the model every x steps.
      --results_dir           (default "")          directory to save results
      --debug_log_filename    (default "debug.log") filename of debugging output
      -b,--batch_size         (default 20)          minibatch size
      --seq_length            (default 20)          unroll length
      --layers                (default 2)           number of hidden layers
      --decay                 (default 2)           the inverse of this decays the learning rate (--lr).
      --rnn_size              (default 200)         hidden unit size
      --dropout               (default 0) 
      --init_weight           (default 0.1)         random weight initialization limits
      --lr                    (default 1)           learning rate
      --vocab_size            (default 10000)       limit on the vocabulary size
      --decay_epoch           (default 4)           when to start decaying learning rate
      --max_epoch             (default 13)          final epoch
      --max_grad_norm         (default 5)           clip when gradients exceed this norm value
   ]]
   return opt
end

function transfer_data(x)
    if gpu then
        return x:cuda()
    else
        return x
    end
end

model = {}

local function lstm(x, prev_c, prev_h)
    -- Calculate all four gates in one go
    local i2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
    local h2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
    local gates            = nn.CAddTable()({i2h, h2h})

    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshaped_gates   =  nn.Reshape(4,params.rnn_size)(gates)
    local sliced_gates     = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return next_c, next_h
end

-- Updated from the original version so that the log-prediction is also returned.
function create_network()
    local x                  = nn.Identity()() -- input word
    local y                  = nn.Identity()() -- word you're trying to predict from x.
    -- prev_c and prev_h are stored in prev_s
    local prev_s             = nn.Identity()() -- previous state
    -- rrn_size -> word embedding size
    local i                  = {[0] = nn.LookupTable(params.vocab_size,
						     params.rnn_size)(x)}
    -- next state of the model
    local next_s             = {}
    -- split c and h
    local split              = {prev_s:split(2 * params.layers)}
    for layer_idx = 1, params.layers do
        local prev_c         = split[2 * layer_idx - 1]
        local prev_h         = split[2 * layer_idx]
        local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
        local next_c, next_h = lstm(dropped, prev_c, prev_h)
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h
    end
    local h2y                = nn.Linear(params.rnn_size, params.vocab_size)
    local dropped            = nn.Dropout(params.dropout)(i[params.layers])
    local pred               = nn.LogSoftMax()(h2y(dropped))
    local err                = nn.ClassNLLCriterion()({pred, y})
    local module             = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s), pred})
    -- initialize weights
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return transfer_data(module)
end

function build_model()
    DEBUG("Creating a RNN LSTM network.")
    local core_network = create_network()
    paramx, paramdx = core_network:getParameters()
    model.s = {}
    model.ds = {}
    model.start_s = {}
    for j = 0, params.seq_length do
        model.s[j] = {}
        for d = 1, 2 * params.layers do
            model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        end
    end
    for d = 1, 2 * params.layers do
        model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
    model.core_network = core_network
    model.rnns = g_cloneManyTimes(core_network, params.seq_length)
    model.norm_dw = 0
    model.err = transfer_data(torch.zeros(params.seq_length))
end

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

function fp(state)
    -- g_replace_table(from, to).  
    g_replace_table(model.s[0], model.start_s)
    
    -- reset state when we are done with one full epoch
    if state.pos + params.seq_length > state.data:size(1) then
        reset_state(state)
    end
    
    -- forward prop
    for i = 1, params.seq_length do
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
	-- NOTE added _ for pred
        model.err[i], model.s[i], _ = unpack(model.rnns[i]:forward({x, y, s}))
        state.pos = state.pos + 1
    end
    
    -- next-forward-prop start state is current-forward-prop's last state
    g_replace_table(model.start_s, model.s[params.seq_length])
    
    -- cross entropy error
    return model.err:mean()
end

-- To nullify the affect of having the network return the prediction, I added a
-- matrix of zero gradients called 'dummy_pred_grad'.
function bp(state)
    -- start on a clean slate. Backprop over time for params.seq_length.
    paramdx:zero()
    reset_ds()
    for i = params.seq_length, 1, -1 do
        -- to make the following code look almost like fp
        state.pos = state.pos - 1
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        -- Why 1?
        local derr = transfer_data(torch.ones(1))
	-- NOTE: added dummy_pred_grad.
	local dummy_pred_grad = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
        -- tmp stores the ds
        local tmp = model.rnns[i]:backward({x, y, s},
                                           {derr, model.ds, dummy_pred_grad})[3]
        -- remember (to, from)
        g_replace_table(model.ds, tmp)
    end
    
    -- undo changes due to changing position in bp
    state.pos = state.pos + params.seq_length
    
    -- gradient clipping
    model.norm_dw = paramdx:norm()
    if model.norm_dw > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / model.norm_dw
        paramdx:mul(shrink_factor)
    end
    
    -- gradient descent step
    paramx:add(paramdx:mul(-params.lr))
end

function run_valid()
    -- again start with a clean slate
    reset_state(state_valid)
    
    -- no dropout in testing/validating
    g_disable_dropout(model.rnns)
    
    -- collect perplexity over the whole validation set
    local len = (state_valid.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    for i = 1, len do
        perp = perp + fp(state_valid)
    end
    DEBUG("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
    g_enable_dropout(model.rnns)
end

-- invert the ptb.vocab_map
-- Used to convert the output of the classifier back into words.
function invert_vocab_map()
   local vocab_idx_map = {}
   for k,v in pairs(ptb.vocab_map) do
      vocab_idx_map[v] = k
   end
   return vocab_idx_map
end

-- Given a 1d tensor of seed words (as indexes to the vocab), this function uses
-- the model to generate 'sentence_length' more words. It uses the multinomial
-- distribution to sample from the predictions so that the results are not
-- deterministic.
function run_sentence_gen(sentence_length, word_idxs)
   -- reset model
   for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
   end

   g_disable_dropout(model.rnns)

   -- put ones where there are no entries. this is equivalent to guessing the first vocab word.
   local sentence_idxs = torch.ones(word_idxs:size(1)+sentence_length)
   for i = 1, word_idxs:size(1) do
      sentence_idxs[i] = word_idxs[i]
   end
   -- Resize and replicate the inputs to match the batch size like data.testdataset does.
   local sentence_inputs = sentence_idxs:resize(sentence_idxs:size(1), 1):expand(sentence_idxs:size(1), params.batch_size)
   
   g_replace_table(model.s[0], model.start_s)
   for i = 1, sentence_inputs:size(1)-1 do
      local x = sentence_inputs[i]
      local y = sentence_inputs[i+1]

      _, model.s[1], log_pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
      -- only save the word if it's one of the newly predicted ones
      if i >= word_idxs:size(1) then
         -- Sampling from the predictions so the results are deterministic.
         local pred_idx = torch.multinomial(torch.exp(log_pred), 1)
	 sentence_inputs[i+1] = pred_idx
      end
      g_replace_table(model.s[0], model.s[1])
   end
   g_enable_dropout(model.rnns)
   return sentence_idxs
end

-- Expected query syntax:
-- len word1 word2 word3 ...
-- 'len' is the length of the desired sentences
-- word${i} are the input words.
-- uses the vocab_idx_map to check that the words are in the vocab.
-- Returns the split line
function read_query()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = string.lower(line)
  line = stringx.split(line)
  local sentence_length = tonumber(line[1])
  if sentence_length == nil then
     error({code="init"})
  end
  word_indices = {}
  for i = 2,#line do
     if ptb.vocab_map[line[i]] == nil then
	error({code="vocab", word = line[i]})
     end
  end
  return line
end

function sentence_gen_repl()
   local vocab_idx_map = invert_vocab_map()
   
    while true do
      DEBUG("Query: len word1 word2 etc")
      local ok, line = pcall(read_query)
      if not ok then
	 if line.code == "EOF" then
	    break -- end loop
	 elseif line.code == "vocab" then
	    DEBUG("Word not in vocabulary: "..line.word)
	 elseif line.code == "init" then
	    DEBUG("Start with a number")
	 else
	    DEBUG(line)
	    DEBUG("Failed, try again")
	 end
      else
	 local sentence_length = tonumber(line[1])
	 local input_idxs = torch.zeros(#line - 1)
	 for i = 2,#line do
	    input_idxs[i-1] = ptb.vocab_map[line[i]]
	 end
	 predicted_word_idxs = run_sentence_gen(sentence_length, input_idxs)
	 all_words_str = ""
	 for i = 1,predicted_word_idxs:size(1) do
	    all_words_str = all_words_str..vocab_idx_map[predicted_word_idxs[i][1]].." "
	 end
	 DEBUG(all_words_str)
      end
   end
end

function save_model(experiment_dir)
   local filename = paths.concat(experiment_dir, 'model.net')
   DEBUG('==> saving model to '..filename)
   torch.save(filename, model)
end

function train_model(experiment_dir)   
   DEBUG("Network parameters:")
   DEBUG(params)
   
   local states = {state_train, state_valid}
   for _, state in pairs(states) do
      reset_state(state)
   end
   
   step = 0
   epoch = 0
   total_cases = 0
   beginning_time = torch.tic()
   start_time = torch.tic()

   DEBUG("Starting training.")
   words_per_step = params.seq_length * params.batch_size
   epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
   
   while epoch < params.max_epoch do  
      if step % 5 == 0 then
         DEBUG('step: '..step)
         DEBUG('epoch: '..epoch)
      end
      -- take one step forward
      perp = fp(state_train)
      if perps == nil then
	 perps = torch.zeros(epoch_size):add(perp)
      end
      perps[step % epoch_size + 1] = perp
      step = step + 1
      
      -- gradient over the step
      bp(state_train)
      
      -- words_per_step covered in one step
      total_cases = total_cases + params.seq_length * params.batch_size
      epoch = step / epoch_size

      -- display details at some interval
      if step % params.model_save_freq == 0 then
	 wps = torch.floor(total_cases / torch.toc(start_time))
	 since_beginning = g_d(torch.toc(beginning_time) / 60)
	 DEBUG('epoch = ' .. g_f3(epoch) ..
		  ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
		  ', wps = ' .. wps ..
		  ', dw:norm() = ' .. g_f3(model.norm_dw) ..
		  ', lr = ' ..  g_f3(params.lr) ..
		  ', since beginning = ' .. since_beginning .. ' mins.')
	 save_model(experiment_dir)
      end
      
      -- run when epoch done
      if step % epoch_size == 0 then
	 run_valid()
	 if epoch > params.decay_epoch then
            params.lr = params.lr / params.decay
	 end
      end
   end
   save_model(experiment_dir)
   DEBUG("Training is over.")
end

-- main code starts here

params = parse_cmdline()
experiment_dir = setup_experiment(params)
-- NOTE: the DEBUG() function now callable

if gpu then
    g_init_gpu(arg)
end

-- get data in batches
-- Training data is loaded in both train and test mode, since it loads the vocab map.
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}

if params.mode == 'query' then
   DEBUG('QUERY MODE')
   DEBUG('Loading model file from '..params.model_file)
   model = torch.load(params.model_file)
   sentence_gen_repl()
else
   DEBUG('TRAIN MODE')
   build_model()
   train_model(experiment_dir)
end
