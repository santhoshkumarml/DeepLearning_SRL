--
-- User: santhosh
-- Date: 11/2/15
--
require 'Constants'
require 'torch'
require 'nn'

local w2vutils = require 'w2vutils'

local ksz = 3
local convOutputFrame = 100
local HUs = 100
--Later loaded to number of argument classes
local final_output_layer_size = -1
local UNK = torch.Tensor(WORD_VEC_SIZE):fill(0)

local total_data_size = 45000
local train_data_size = math.floor(0.7 * total_data_size)
local test_data_size = total_data_size - train_data_size
global_net = {}

function init_nn(isLoad)
    local f = io.open(SRL_TEMPORAL_NET_FILE)
    if isLoad and f~=nil then
        global_net = torch.load(SRL_TEMPORAL_NET_FILE)
        if f~=nil then f:close() end
    else
        global_net = nn.Sequential()
        global_net:add(nn.TemporalConvolution(WORD_VEC_SIZE
                + SRL_WORD_INTEREST_DIST_DIM + SRL_VERB_DIST_DIM,
            convOutputFrame, ksz))
        --    lolca sentence_size = sentence_size - (2 * math.floor(ksz/2))
        global_net:add(nn.TemporalMaxPooling(1))
        global_net:add(nn.Tanh())
        global_net:add(nn.Linear(HUs, final_output_layer_size))
        global_net:add(nn.LogSoftMax())
    end
end

function update_nn_for_sentence(sentence)
    local sentence_size = sentence:size(1)
    sentence_size = sentence_size - (2 * math.floor(ksz/2))
    global_net.modules[2] = nn.TemporalMaxPooling(sentence_size)
end

--Serialize the neural net
function save_nn()
    torch.save(SRL_TEMPORAL_NET_FILE, global_net)
end


--Encode the number in 5 dimenion space
--Also the number is restrained to stick to [-15, 15] if it is not
function intToBin(num)
    local isNeg = false
    if num > 15 then num = 15 else if num < -15 then num = -15 end end
    if num < 0 then isNeg = true end
    local num = math.abs(num)
    local tensor = torch.Tensor(5)
    local start_idx = 5
    while num > 0 do
        local div = (num % 2)
        num = math.floor(num / 2)
        tensor[start_idx] = div
        start_idx = start_idx - 1
    end
    if isNeg then tensor[1] = 1 else tensor[1] = 0 end
    return tensor
end


--Read Arguments dictionary and generate class number for them starting with 1
function makeArgToClassDict()
    local f = io.open(ARGS_FILE)
    local args = string.split(f:read(), ",")
    local arg_ds = {}
    local arg_to_class_dict, class_to_arg_dict = {}, {}
    for idx = 1, #args do
        class_to_arg_dict[idx] = args[idx]
        arg_to_class_dict[args[idx]] = idx
    end
    arg_ds[1] = arg_to_class_dict
    arg_ds[2] = class_to_arg_dict
    torch.save(ARGS_DICT_FILE, {arg_to_class_dict, class_to_arg_dict})
    f:close()
    return #args
end


--Train on an instance of sentences with a specific word of interest.
function trainForSingleInstance(train_data)
    local sentence = train_data[1][1]
    update_nn_for_sentence(sentence)
    local criterion = nn.ClassNLLCriterion()
    local trainer = nn.StochasticGradient(global_net, criterion)
    trainer.learningRate = 0.01
    trainer.maxIteration = 1
    trainer:train(train_data)
    save_nn()
end

--Train for sentences
function train(epoch, epoch_checkpt, sent_checkpt)
    print('--------------------------Train iteration number:'..epoch..'----------------------------------------')
    -- load data structures for class_to_arg_name conversion and arg_name_to_class conversion
    local arg_ds = torch.load(ARGS_DICT_FILE)
    local arg_to_class_dict, class_to_arg_dict = arg_ds[1], arg_ds[2]
    local f = io.open(SRL_TRAIN_FILE)
    local current_run = 0

    for sent_num = 1, train_data_size do
        local predicate_idx = tonumber(f:read())
        local words = string.split(f:read(), " ")
        local args = string.split(f:read(), " ")
        print('Processing the sentence', sent_num)
        if epoch > epoch_checkpt or sent_num > sent_checkpt then
            for widx1 = 1, #words do
                local word_of_interest, current_arg = words[widx1], args[widx1]
                local feature_vecs_for_sent = torch.Tensor(#words + 2, WORD_VEC_SIZE
                        + SRL_WORD_INTEREST_DIST_DIM + SRL_VERB_DIST_DIM):fill(0)
                for widx2 = 1, #words do
                    local curr_word = words[widx2]
                    local feature_vec_for_word = w2vutils:word2vec(curr_word)
                    if not feature_vec_for_word then
                        feature_vec_for_word = UNK
                        --print('Word Vec not known for', curr_word)
                    else
                        feature_vec_for_word = feature_vec_for_word:narrow(1, 1, WORD_VEC_SIZE)
                    end

                    --Convert distance to binary tensor and append it to word vector
                    local distance_to_word_of_interest = intToBin(widx1 - widx2)
                    local distance_to_predicate = intToBin(predicate_idx - widx2)
                    local feature_vec = torch.cat(
                        torch.cat(feature_vec_for_word, distance_to_word_of_interest),
                        distance_to_predicate)
                    feature_vecs_for_sent[widx2 + 1] = feature_vec
                end

                local curr_target = arg_to_class_dict[current_arg]
                local train_data = {}
                train_data[1] = {feature_vecs_for_sent, curr_target}
                function train_data:size() return 1 end
                trainForSingleInstance(train_data)
                feature_vecs_for_sent:free()
                collectgarbage()
            end
            if current_run == 100 then
                save_nn()
                local checkPt = {epoch, sent_num}
                torch.save(SRL_CHECKPT_FILE, checkPt)
                current_run = 0
            else current_run = current_run + 1 end
        end
    end
    save_nn()
    f:close()
    print('------------------------------------------------------------------------------------------')
end


-- Remove the serialized nets
function doCleanup()
    os.remove(SRL_TEMPORAL_NET_FILE)
end

--Main Function
function main()
    --doCleanup()
    local checkPt = {1, 1}
    local f = io.open(SRL_CHECKPT_FILE)
    if f~=nil then
        checkPt = torch.load(SRL_CHECKPT_FILE)
        f:close()
    end
    local epoch_checkpt = checkPt[1]
    local sent_checkpt = checkPt[2]
    --Number of different argument classes
    final_output_layer_size = makeArgToClassDict()
    init_nn(true)
    for epoch = epoch_checkpt, EPOCH do
        train(epoch, epoch_checkpt, sent_checkpt)
    end
end
main()