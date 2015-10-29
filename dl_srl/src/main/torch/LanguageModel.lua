require 'torch';
require 'nn';

WORDS_FILE_PATH = "../resources/diction.txt"

TRAIN_DATA_FILE_PATH = "../resources/train_data.txt"

DICTIONARY_FILE = "../resources/dictionary.dict"

WORD_VEC_SIZE = 25

WINDOW_SIZE = 5

--will take BATCH_SIZE +ve and BATCH_SIZE -ve samples
BATCH_SIZE = 5

EPOCH = 1

START = "$START$"

FINISH = "$END$"



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
        table.insert(words, START)
        for word in l:gmatch("%w+") do 
            table.insert(words, word)
            if not word_dict[word] then
                word_dict[word] = torch.randn(WORD_VEC_SIZE)
            else
            end
        end
    end
    torch.save(DICTIONARY_FILE, word_dict)
end
   
function construct_nn(window_size, word_vec_size, hidden_layer_nodes)
    -- Add NN Layers
    local net = nn.Sequential()

    net:add(nn.Linear(window_size * word_vec_size, hidden_layer_nodes))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(hidden_layer_nodes, 2))
    net:add(nn.LogSoftMax())

    return net
end

function readBatchData(f, word_dict)
	local cnt_train_data = 1

	local window_words = {}

	local batch_train = {}

    while cnt_train_data <= BATCH_SIZE do
        local train_data = {}

        local pl = f:read()
        local nl = f:read()

        if not pl or not nl then break end

        local pos_words = {}
        local neg_words = {}

        local start_idx = 1
        local end_idx = WORD_VEC_SIZE

        for word in pl:gmatch("%w+") do 
            pos_word_vec = word_dict[word]
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

        for word in nl:gmatch("%w+") do 
        	neg_word_vec = word_dict[word]
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

    function batch_train:size() return cnt_train_data * 2 end

    return  batch_train, window_words
end


function trainAndUpdatedWordVec(net, epoch)
    for e = 1, epoch do
    	local f = io.open(TRAIN_DATA_FILE_PATH)
    	while true do
    		word_dict = torch.load(DICTIONARY_FILE)

	    	batch_train_data, window_words = readBatchData(f, word_dict)

	    	if not batch_train_data:size(0) == 0 then break end

	    	-- Define Loss Function
    		local criterion = nn.ClassNLLCriterion()

			local trainer = nn.StochasticGradient(net, criterion)
			trainer.learningRate = 0.01
            trainer.maxIteration = 1

			trainer:train(batch_train_data)

			for i = 1, #window_words do
				local words = window_words[i]
                local start_idx = 1
                local end_idx = WORD_VEC_SIZE
				for w = 1, # words do 
					local word_vec = word_dict[words[w]]
                    local gradIp = net.gradInput
                    for idx = start_idx, end_idx do 
        			     word_vec[idx - start_idx + 1] = word_vec[idx - start_idx + 1] - word_vec[idx - start_idx + 1] * net.gradInput[idx]
                    end
        			word_dict[words[w]] = word_vec
                    start_idx = end_idx
                    end_idx = start_idx + WORD_VEC_SIZE - 1
        		end 
        	end
        	torch.save(DICTIONARY_FILE, word_dict)
	    end
    end
end

saveWordVecForWordsInDict()
net = construct_nn(WINDOW_SIZE, WORD_VEC_SIZE, 50)
trainAndUpdatedWordVec(net, EPOCH)