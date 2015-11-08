--
-- User: santhosh
-- Date: 10/27/15
-- Language Model
--

require 'torch';
require 'nn';
require 'Constants';

-- create/update and store word vectors for dictionary.
function initOrUpdateWordVecForWordsInDict(netGradIp)
    local f = io.open(WORDS_FILE_PATH)
    local isUpdate = true
    local word_dict = {}

    if not netGradIp then
        isUpdate = false
        word_dict[START] = torch.randn(WORD_VEC_SIZE)
        word_dict[FINISH] = torch.randn(WORD_VEC_SIZE)
    else
        word_dict = torch.load(DICTIONARY_FILE)
    end

    -- netGradIp will be WORD_VEC_SIZE * WINDOWS_SIZE
    -- To update a single word we will need to just pick the gradient of weights For Middle word
    -- (i.e ((WINDOW_SIZE/2) + 1)* WORD_VEC_SIZE) to ((WINDOW_SIZE/2) + 1)* WORD_VEC_SIZE) + WORD_VECSIZE)
    -- and update the word vector by word_vec = word_vec - word_vec * (above gradWeights)
    local gradIpOffset = ((math.floor(WINDOW_SIZE / 2) + 1)* WORD_VEC_SIZE)
    while true do
        local word = f:read()
        if not word then break end
        if not isUpdate then
            word_dict[word] = torch.randn(WORD_VEC_SIZE)
        else
            word_vec = word_dict[word]
            for idx = 1, WORD_VEC_SIZE do
                word_vec[idx] = word_vec[idx] - word_vec[idx] * netGradIp[gradIpOffset + idx]
            end
            word_dict[word] = word_vec
        end
    end
    torch.save(DICTIONARY_FILE, word_dict)
end

-- load NN from the file system if present or create a new one.
function get_or_construct_nn()
    local f = io.open(LANGUAGE_NET_FILE)
    local net = {}
    if not f then
        net = nn.Sequential()
        -- Add NN Layers
        local inputs = WINDOW_SIZE * WORD_VEC_SIZE;
        local outputs = 2;
        local HUs = 50;

        net:add(nn.Linear(inputs, HUs))
        net:add(nn.Sigmoid())
        net:add(nn.Linear(HUs, outputs))
        net:add(nn.LogSoftMax())
    else
        net = torch.load(LANGUAGE_NET_FILE)
    end
    return net
end

-- Read a small batch of data for training.
function readBatchData(f)
	local cnt_train_data = 1
	local batch_train = {}
	word_dict = torch.load(DICTIONARY_FILE)

    while cnt_train_data <= BATCH_SIZE do
        local train_data = {}
        local pl = f:read()
        local nl = f:read()

        if not pl or not nl then break end

        local pos_words = {}
        local neg_words = {}

        local start_idx = 1
        local end_idx = WORD_VEC_SIZE

        pl_split = string.split(pl," ")
        nl_split = string.split(nl," ")

        for w_idx =1, #pl_split do
            local word = pl_split[w_idx]
            local pos_word_vec = word_dict[word]
            for idx = start_idx, end_idx do 
                train_data[idx] = pos_word_vec[idx - start_idx + 1]
            end
            table.insert(pos_words, word)
            start_idx = end_idx + 1
            end_idx = start_idx + WORD_VEC_SIZE - 1
        end

        batch_train[2 * cnt_train_data - 1] = {torch.Tensor(train_data), 1}

        start_idx = 1
        end_idx = WORD_VEC_SIZE

        for w_idx =1, #nl_split do
            local word = nl_split[w_idx]
        	local neg_word_vec = word_dict[word]
            for idx = start_idx, end_idx do 
                train_data[idx] = neg_word_vec[idx - start_idx + 1]
            end
            table.insert(neg_words, word)
            start_idx = end_idx + 1
            end_idx = start_idx + WORD_VEC_SIZE - 1
        end

        batch_train[2 * cnt_train_data] = {torch.Tensor(train_data), 2}

        cnt_train_data = cnt_train_data + 1
    end

    cnt_train_data = cnt_train_data - 1

    function batch_train:size() return cnt_train_data * 2 end

    return  batch_train
end


-- Train an online using small batch data and update the word vectors.
function trainAndUpdatedWordVec(epoch)
    -- Define Loss Function
    local net = get_or_construct_nn()
    local criterion = nn.ClassNLLCriterion()
    local trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.01
    trainer.maxIteration = 1

    for e = 1, epoch do
        print('Starting iteration:', e)
    	local f = io.open(TRAIN_DATA_FILE_PATH)
        -- Run Batch Training and Gradient descend
    	while true do
            batch_train_data = readBatchData(f)
	    	if batch_train_data:size() == 0 then break end
			trainer:train(batch_train_data)
        end

        -- Update Word Vector and the network
        initOrUpdateWordVecForWordsInDict(net.gradInput)
        torch.save(LANGUAGE_NET_FILE, net)
    end
end

--Main Function
function main()
    initOrUpdateWordVecForWordsInDict()
    trainAndUpdatedWordVec(EPOCH)
end

main()