--
-- User: santhosh
-- Date: 11/30/15
--
local test_sent_start = 4413
local test_sent_end = 5413
local w2vutils = require 'w2vutils'
local UNK = torch.Tensor(WORD_VEC_SIZE):fill(0)
local global_net = {}

function main()
    init_nn(true)
    local accuracy = test_SRL()
    print(accuracy)
end

main()