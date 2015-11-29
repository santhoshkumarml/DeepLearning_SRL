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

--Load the Temporal Convolution layer for the neural network
function get_temporal_nn_for_srl()
    local f = io.open(SRL_TEMPORAL_NET_FILE)
    local temporal_net = nn.Sequential()
    local temporal_convolution = {}
    if not f then
        temporal_convolution = nn.TemporalConvolution(WORD_VEC_SIZE
                + SRL_WORD_INTEREST_DIM + SRL_VERB_DIST_DIM,
            convOutputFrame, ksz)
    else
        temporal_convolution = torch.load(SRL_TEMPORAL_NET_FILE)
    end
    temporal_net:add(temporal_convolution)
    return temporal_net
end

--Load the constant layers for the neural network
function get_constant_nn_after_polling()
    local f = io.open(SRL_CONSTANT_NET_FILE)
    local constant_layers = {}
    if not f then
        local constant_layers = {}
        constant_layers[1] = nn.Tanh()
        constant_layers[2] = nn.Linear(HUs, final_output_layer_size)
        constant_layers[3] = nn.LogSoftMax()
    else
        constant_layers = torch.load(SRL_CONSTANT_NET_FILE)
    end
    return constant_layers
end

--Load the neural network for sentence
function get_nn_for_sentence(sentence)
    local net = get_temporal_nn_for_srl()
    local constant_layers = get_constant_nn_after_polling()
    local sentence_size = sentence:size(1)

    sentence_size = sentence_size - (2 * math.floor(ksz/2))

    net:add(nn.TemporalMaxPooling(sentence_size))

    for i = 1, #constant_layers do
        net:add(constant_layers[i])
    end
    return net, constant_layers
end

--Serialize the neural net
function save_nn(net, constant_layers)
    --Save the first module(i.e Temporal Convolution)
    torch.save(SRL_TEMPORAL_NET_FILE, net:get(1))
    --Save the constant layer modules
    torch.save(SRL_CONSTANT_NET_FILE, constant_layers)
end


--Train on an instance of sentences with a specific word of interest.
function trainForSingleInstance(sentence)
    local net = get_nn_for_sentence(sentence)
    local criterion = nn.ClassNLLCriterion()
    local trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.01
    trainer.maxIteration = 1
    trainer:train(train_data)
    save_nn(net)
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
        print(num, div)
        tensor[start_idx] = div
        start_idx = start_idx - 1
    end
    if isNeg then tensor[1] = 1 else tensor[1] = 0 end
    return tensor
end

-- Remove the serialized nets
function doCleanup()
    os.remove(SRL_TEMPORAL_NET_FILE)
    os.remove(SRL_CONSTANT_NET_FILE)
end


--Read Arguments dictionary and generate class number for them starting with 1
function makeArgToClassDict()
    local f = io.open(ARG_FILE)
    local args = string.split(f:read(), ",")
    local arg_to_class_dict, class_to_arg_dict = {}, {}
    for idx = 1, #args do
        class_to_arg_dict[idx] = args[idx]
        arg_to_class_dict[args[idx]] = idx
    end
    torch.save(ARGS_DICT_FILE, {arg_to_class_dict, class_to_arg_dict})
    return #args
end

--Main Function
function main()
    doCleanup()
    --Number of different argument classes
    final_output_layer_size = makeArgToClassDict()
    -- load data structures for class_to_arg_name conversion and arg_name_to_class conversion
    local arg_to_class_dict, class_to_arg_dict = torch.load(ARGS_DICT_FILE)

    --TODO: read this from file
    local sentences = {}
    for iter = 1, #sentences do
        local sentence = sentences[iter]
        local wordsAndArgs = string.split(sentence, " ")
        local feature_vecs_for_sent = torch.Tensor(#wordsAndArgs, WORD_VEC_SIZE
                + SRL_WORD_INTEREST_DIST_DIM + SRL_VERB_DIST_DIM)
        --TODO: Fill this from the file the first line is predicate index
        local predicate_idx = -1
        for widx1 = 1, #wordsAndArgs do
            local word_of_interest, current_arg = string.split(wordsAndArgs[widx1], ",")
            for widx2 = 1, #wordsAndArgs do
                local curr_word, arg = string.split(wordsAndArgs[widx2], ",")
                local feature_vec_for_word = w2vutils:word2vec(curr_word)
                feature_vec_for_word = feature_vec_for_word:narrow(1, 1, WORD_VEC_SIZE)

                --Convert distance to binary tensor and append it to word vector
                local distance_to_word_of_interest = intToBin(widx1 - widx2)
                local distance_to_predicate = intToBin(predicate_idx - widx2)
                feature_vec_for_word = torch.cat(
                    torch.cat(feature_vec_for_word, distance_to_word_of_interest),
                    distance_to_predicate)
                feature_vecs_for_sent[widx2] = feature_vec_for_word
            end

            local curr_target = arg_to_class_dict[current_arg]
            local train_data = {}
            train_data[1] = {feature_vecs_for_sent, curr_target}
            function train_data:size() return 1 end
            trainForSingleInstance(train_data)
        end
    end
end


function sample_test_sentence(sentence_size)
    local train_sentence = torch.Tensor(sentence_size, WORD_VEC_SIZE
            + SRL_WORD_INTEREST_DIST_DIM + SRL_VERB_DIST_DIM)
    return train_sentence
end


main()