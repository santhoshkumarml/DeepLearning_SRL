--
-- User: santhosh
-- Date: 11/2/15
--
require 'Constants'
require 'torch'
require 'nn'

function get_temporal_nn_for_srl(sentence_size)
     -- Add NN Layers
    local temporal_net = nn.Sequential()
    local ksz = 3
    local convOutputFrame = 100
    temporal_net:add(nn.TemporalConvolution(WORD_VEC_SIZE, convOutputFrame, ksz))
    temporal_net:add(nn.TemporalMaxPooling(sentence_size))
    return temporal_net
end


function get_constant_nn_after_polling()
    local HUs = 100
    local outputs = 2
    local constant_layers = {}
    constant_layers[1] = nn.Tanh()
    constant_layers[2] = nn.Linear(HUs, outputs)
    constant_layers[3] = nn.LogSoftMax()
    return constant_layers
end

function trainForSentence(sentence, constant_layers)
    local sentence_size = sentence:size(1)
    local net = get_temporal_nn_for_srl(sentence_size)
    for i = 1, #constant_layers do
        net:add(constant_layers[i])
    end

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

function sample_test_sentence()
    local sentence_size = 5
    local t1 = torch.Tensor(sentence_size + 2, WORD_VEC_SIZE):fill(1)
    t1[1]:fill(0)
    t1[sentence_size + 2]:fill(0)
    return t1
end

function main()
    local constant_layers = get_constant_nn_after_polling()
    local train_data = sample_test_sentence()
    trainForSentence(train_data, constant_layers)
end

main()