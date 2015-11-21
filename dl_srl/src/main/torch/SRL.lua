--
-- User: santhosh
-- Date: 11/2/15
--
require 'Constants'
require 'torch'
require 'nn'

function get_temporal_nn_for_srl()
    local f = io.open(SRL_TEMPORAL_NET_FILE)
    local temporal_net = nn.Sequential()
    local temporal_convolution = {}
    if not f then
        local ksz = 3
        local convOutputFrame = 100
        temporal_convolution = nn.TemporalConvolution(WORD_VEC_SIZE, convOutputFrame, ksz)
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
        local HUs = 100
        local outputs = 2
        local constant_layers = {}
        constant_layers[1] = nn.Tanh()
        constant_layers[2] = nn.Linear(HUs, outputs)
        constant_layers[3] = nn.LogSoftMax()
    else
        constant_layers = torch.load(SRL_CONSTANT_NET_FILE)
    end
    return constant_layers
end

function trainForSentence(sentence, constant_layers)

    local net = get_temporal_nn_for_srl()
    local constant_layers = get_constant_nn_after_polling()
    local sentence_size = sentence:size(1)

    local ksz = 3
    sentence_size = sentence_size - (2 * math.floor(ksz/2))

    net:add(nn.TemporalMaxPooling(sentence_size))

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

    local temporal_net = net:get(1)
    torch.save(SRL_TEMPORAL_NET_FILE, temporal_net)
    torch.save(SRL_CONSTANT_NET_FILE, constant_layers)
end

function sample_test_sentence(sentence_size)
    local t1 = torch.Tensor(sentence_size + 2, WORD_VEC_SIZE)
--    t1[1]:fill(0)
--    t1[sentence_size + 2]:fill(0)
    return t1
end

function doCleanup()
    -- remove the serialized net
    os.remove(SRL_TEMPORAL_NET_FILE)
    os.remove(SRL_CONSTANT_NET_FILE)
end

function main()
    doCleanup()
    local train_data1 = sample_test_sentence(5)
    trainForSentence(train_data1)
    local train_data2 = sample_test_sentence(7)
    trainForSentence(train_data2)
end

main()