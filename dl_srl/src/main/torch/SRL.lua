--
-- User: santhosh
-- Date: 11/2/15
--
require 'Constants'
require 'torch'
require 'nn'

local ksz = 3
local convOutputFrame = 100
local HUs = 100
--change it to number of arguments
local final_output_layer_size = 2

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

function save_nn(net, constant_layers)
    --Save the first module(i.e Temporal Convolution)
    torch.save(SRL_TEMPORAL_NET_FILE, net:get(1))
    --Save the constant layer modules
    torch.save(SRL_CONSTANT_NET_FILE, constant_layers)
end

function trainForSentence(sentence)
    --TODO: put in negative sentences
    local train_data = {}
    train_data[1] = {sentence, 1}
    function train_data:size() return 1 end

    -- Define Loss Function
    local criterion = nn.ClassNLLCriterion()
    local trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.01
    trainer.maxIteration = 1
    trainer:train(train_data)
end

function sample_test_sentence(sentence_size)
    local train_sentence = torch.Tensor(sentence_size + 2, WORD_VEC_SIZE
            + SRL_WORD_INTEREST_DIST_DIM + SRL_VERB_DIST_DIM)
    train_sentence[1]:fill(0)
    train_sentence[sentence_size + 2]:fill(0)
    return train_sentence
end

function doCleanup()
    -- remove the serialized nets
    os.remove(SRL_TEMPORAL_NET_FILE)
    os.remove(SRL_CONSTANT_NET_FILE)
end

function main()
    doCleanup()
    for iter = 1, 2 do
        local train_data = sample_test_sentence(math.random(5, 12))
        trainForSentence(train_data)
    end
end

main()