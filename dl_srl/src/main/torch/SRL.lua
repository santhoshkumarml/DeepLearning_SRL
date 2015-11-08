--
-- User: santhosh
-- Date: 11/2/15
--
require 'Constants';

function get_nn_for_srl(sentence_size)
     -- Add NN Layers
    local net = nn.Sequential()

    local convInputFrame = 3;
    local convOutputFrame = 100;
    local outputs = 2;
    local HUs = 100;
    local ksz = 3;

    net:add(nn.TemporalConvolution(convInputFrame, convOutputFrame, ksz))
    net:add(nn.TemporalMaxPooling(conVHUs, HUs))
    net:add(nn.Tanh())
    net:add(nn.Linear(HUs, outputs))
    net:add(nn.LogSoftMax())
    return net
end
