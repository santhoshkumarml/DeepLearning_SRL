require 'torch';
require 'nn';

local FILE_PATH = "../resources/diction.txt"
local WORD_VEC_SIZE = 25
local DICTIONARY_FILE = "../resources/dictionary.dict"


-- read and Store Data
function readFileAndCreateDictorinary(file_path)
    local f = io.open(file_path)
    local store = {}
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
        table.insert(words,FINISH)
        table.insert(store, words)
    end
    torch.save(DICTIONARY_FILE, word_dict)
    return store
end

-- split into windows and make training data
function makeTrainingData(store, window_size, word_dict)
    local train_data = {}
    for idx, line in ipairs(store) do
        local size = table.getn(line)
        local no_of_windows = math.max( 0, size - window_size) + 1

        for widx = 1, no_of_windows do
            local training_inst = {}
            local data = {}
            local word_idx = widx

            --Add to data the words for the window
            while table.getn(data) < window_size and word_idx < size do
                table.insert(data, line[word_idx])
                word_idx = word_idx + 1
            end
                
            --Align to Window Size
            while word_idx >= size and table.getn(data) < window_size do
                table.insert(data, FINISH)
                word_idx = word_idx + 1
            end

            -- Get WordVector From Dictionary
            --local word_vec_data = word_dict[data[1]]
            --for i = 2, table.getn(data) do
            --    word_vec_data = torch.cat(word_vec_data, word_dict[data[i]])
            --end
            
            training_inst.data = word_vec_data
            training_inst.label = 1

            table.insert(train_data, training_inst)
        end   
    end
    return train_data
end
   
function construct_nn(window_size, word_vec_size, hidden_layer_nodes)
    -- Add NN Layers
    local net = nn.Sequential()
    net:add(nn.Linear(window_size * word_vec_size, hidden_layer_nodes))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(hidden_layer_nodes, 1))
    net:add(nn.LogSoftMax())
    -- Define Loss Function
    local criterion = nn.ClassNLLCriterion()
    return net, criterion
end

function trainAndUpdatedWordVec(net, criterion, epoch, input, output)
    for e = 1, epoch do
        -- feed it to the neural network and the criterion
        criterion:forward(net:forward(input), output)

        -- train over this example in 3 steps
        -- (1) zero the accumulation of the gradients
        net:zeroGradParameters()

        -- (2) accumulate gradients
        net:backward(input, criterion:backward(net.output, output))
        
        print(net.gradInput)

        input = input - input*net.gradInput
        
        -- (3) update parameters with a 0.01 learning rate
        net:updateParameters(0.01)
    end
end

function readBatchData(f, word_dict)
    local cnt_train_data = 0

    local window_words = {}

    local batch_train_data = {}
    local batch_train_label = {}
    local batch_train = {}

    while cnt_train_data < BATCH_SIZE do
        local pl = f:read()
        local nl = f:read()
        if not pl or not nl then break end

        local pos_word_vec = nil
        local neg_word_vec = nil
        local pos_words = {}
        local neg_words = {}

        for word in pl:gmatch("%w+") do 
            if not pos_word_vec then
                pos_word_vec = word_dict[word]
            else
                pos_word_vec = torch.cat(pos_word_vec, word_dict[word])
            end
            table.insert(pos_words, word)
        end

        for word in nl:gmatch("%w+") do 
            if not neg_word_vec then
                neg_word_vec = word_dict[word]
            else
                neg_word_vec = torch.cat(neg_word_vec, word_dict[word])
            end
            table.insert(neg_words, word)
        end

        table.insert(batch_train_data, pos_word_vec)
        table.insert(batch_train_label, 1)

        table.insert(batch_train_data, neg_word_vec)
        table.insert(batch_train_label, 0)

        table.insert(window_words, pos_words)
        table.insert(window_words, neg_words)

        cnt_train_data = cnt_train_data + 1
    end

    print(batch_train_data)
    
    batch_train.input = torch.Tensor(batch_train_data)
    batch_train.output = torch.Tensor(batch_train_label)

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

-- store = readFileAndCreateDictorinary(FILE_PATH)
-- word_dict = torch.load(DICTIONARY_FILE)
-- train_data = makeTrainingData(store, WINDOW_SIZE, word_dict)
--net, crit = construct_nn(WINDOW_SIZE, WORD_VEC_SIZE, 50)
--for idx, ins in ipairs(train_data) do
--   data = trainAndUpdatedWordVec(net, crit, ins.data, ins.label)
--    os.exit()
--end