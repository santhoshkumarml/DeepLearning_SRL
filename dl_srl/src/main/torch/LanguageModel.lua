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
    net:add(nn.Linear(hidden_layer_nodes, 1))
    net:add(nn.LogSoftMax())

    return net
end

function readBatchData(f, word_dict)
	local cnt_train_data = 1

	local window_words = {}

	local batch_train_data = torch.Tensor(BATCH_SIZE * 2, WORD_VEC_SIZE * WINDOW_SIZE)
	local batch_train_label = torch.Tensor(BATCH_SIZE * 2)
	local batch_train = {}

    while cnt_train_data <= BATCH_SIZE do
        local pl = f:read()
        local nl = f:read()

        if not pl or not nl then break end

        local pos_words = {}
        local neg_words = {}

        local start_idx = 1
        local end_idx = WORD_VEC_SIZE

        for word in pl:gmatch("%w+") do 
            pos_word_vec = word_dict[word]
            batch_train_data[{2 * cnt_train_data - 1, {start_idx, end_idx}}] = pos_word_vec
            batch_train_label[2 * cnt_train_data-1] = 1
            table.insert(pos_words, word)
            start_idx = end_idx + 1
            end_idx = start_idx + WORD_VEC_SIZE - 1
        end

        start_idx = 1
        end_idx = WORD_VEC_SIZE

        for word in nl:gmatch("%w+") do 
        	neg_word_vec = word_dict[word]
            batch_train_data[{2 * cnt_train_data, {start_idx, end_idx}}] = neg_word_vec
            batch_train_label[2 * cnt_train_data] = 0
            table.insert(neg_words, word)
            start_idx = end_idx + 1
            end_idx = start_idx + WORD_VEC_SIZE - 1
        end

        table.insert(window_words, pos_words)
        table.insert(window_words, neg_words)

        cnt_train_data = cnt_train_data + 1
    end

    batch_train.input = batch_train_data
    batch_train.output = batch_train_label

    print(batch_train_data)
    print(batch_train_label)

    return  batch_train, window_words
end


function trainAndUpdatedWordVec(net, epoch)
    for e = 1, epoch do
    	local f = io.open(TRAIN_DATA_FILE_PATH)
    	while true do
    		word_dict = torch.load(DICTIONARY_FILE)

	    	train_data, window_words = readBatchData(f, word_dict)

	    	if not train_data then break end

	    	-- Define Loss Function
    		local criterion = nn.ClassNLLCriterion()

			local trainer = nn.StochasticGradient(net, criterion)
			trainer.learningRate = 0.01
			trainer:train(dataset)

			print(net.gradInput)

			for i in 1, #window_words do
				words = window_words[i]
				for w in 1, # words do 
					word_vec = word_dict[words[w]]
        			word_vec = word_vec - word_vec * net.gradInput
        			word_dict[words[w]] = word_vec
        		end 
        	end
        	torch.save(DICTIONARY_FILE, word_dict)
	    end
    end
end

saveWordVecForWordsInDict()
net = construct_nn(WINDOW_SIZE, WORD_VEC_SIZE, 50)
trainAndUpdatedWordVec(net, EPOCH)