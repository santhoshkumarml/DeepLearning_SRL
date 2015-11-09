require 'torch';
require 'nn';
require 'Constants';


-- create and store word vectors for dictionary
function saveWordVecForWordsInDict()
    local f = io.open(WORDS_FILE_PATH)
    local word_dict = {}
    word_dict[START] = torch.randn(WORD_VEC_SIZE)
    word_dict[FINISH] = torch.randn(WORD_VEC_SIZE)
    while true do
        local l = f:read()
        if not l then break end
        local words = {}
        -- Why this? ---redundant insertion
        table.insert(words, START)
        word = l
        print(word)
        table.insert(words, word)
        if not word_dict[word] then
            word_dict[word] = torch.randn(WORD_VEC_SIZE)
        else
        end
    end
    --torch.save(DICTIONARY_FILE, word_dict)
    return word_dict
end
   
function construct_nn()
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

function readBatchData(f, word_dict)
	local cnt_train_data = 1
	local window_words = {}
	local batch_train = {}
	--word_dict = torch.load(DICTIONARY_FILE)

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
            -- See if optimization could be done here. [ Do you need to copy word vec to another DS?
            -- What if you could use just the start and end Index needed for the copying later?
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

        table.insert(window_words, pos_words)
        table.insert(window_words, neg_words)

        cnt_train_data = cnt_train_data + 1
    end

    cnt_train_data = cnt_train_data - 1

    -- Why do you need this?
    function batch_train:size() return cnt_train_data * 2 end

    return  batch_train, window_words
end


function trainAndUpdatedWordVec(net, epoch, word_dict)
    -- Define Loss Function
    local criterion = nn.ClassNLLCriterion()
    local trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.01
    trainer.maxIteration = 1

    for e = 1, epoch do
        print('Starting iteration:', e)
    	local f = io.open(TRAIN_DATA_FILE_PATH)
    	while true do
            batch_train_data, window_words = readBatchData(f, word_dict)
	    	if batch_train_data:size() == 0 then break end
			trainer:train(batch_train_data)
        end
        print (net.gradInput, net.gradInput:size(), net.gradInput:nDimension())
        print ("------------")
        for word, word_vec in ipairs(word_dict) do
            print (word)
            print (word_vec:size(), word_vec:nDimension())
            word_vec = word_vec - word_vec * net.gradInput
            word_dict[word] = word_vec
        end
        --torch.save(DICTIONARY_FILE, word_dict)
        break
    end
    -- save net
end

word_dict = saveWordVecForWordsInDict()
net = construct_nn()
trainAndUpdatedWordVec(net, EPOCH, word_dict)