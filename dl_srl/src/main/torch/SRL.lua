--
-- User: santhosh
-- Date: 11/2/15
--
require 'Constants';

function get_nn_for_srl()
     -- Add NN Layers
    local net = nn.Sequential()

    local inputs = WINDOW_SIZE * WORD_VEC_SIZE;
    local outputs = 2;
    local HUs = 50;

    net:add(nn.Linear(inputs, HUs))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(HUs, outputs))
    net:add(nn.LogSoftMax())
    return net
end
