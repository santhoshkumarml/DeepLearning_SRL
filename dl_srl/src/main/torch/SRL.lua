--
-- User: santhosh
-- Date: 11/2/15
--
require 'Constants';

function get_nn_for_srl(sentence_size)
     -- Add NN Layers
    local net = nn.Sequential()

    local inputs = sentence_size * WORD_VEC_SIZE;
    local outputs = 2;
    local HUs = 100;
    local conVHUs = 100;
    local maxPoolingOutput = 3;

    net:add(nn.TemporalConvolution(inputs, conVHUs))
    net:add(nn.TemporalMaxPooling(conVHUs, maxPoolingOutput))
    net:add(nn.Tanh())
    net:add(nn.Linear(HUs, outputs))
    net:add(nn.LogSoftMax())
    return net
end
