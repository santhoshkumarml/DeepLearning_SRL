--
-- Created by IntelliJ IDEA.
-- User: santhosh
-- Date: 11/2/15
-- Time: 10:50 PM
-- To change this template use File | Settings | File Templates.
--
require 'Constants';

function construct_SRL_NN()
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
