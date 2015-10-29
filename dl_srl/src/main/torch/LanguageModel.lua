require 'torch';
require 'nn';

START = "$START$"
FINISH = "$END$"
FILE_PATH = "dataset/wikipedia2text-extracted.txt"
FILE_PATH = "dataset/sample.data"
WORD_VEC_SIZE = 25
WINDOW_SIZE = 5
DICTIONARY_FILE = "dataset/dictionary.dict"


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
            local word_vec_data = word_dict[data[1]]
            for i = 2, table.getn(data) do
                word_vec_data = torch.cat(word_vec_data, word_dict[data[i]])
            end
            
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

function trainAndUpdatedWordVec(net, criterion, input, output)
    for i = 1, 5 do
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
    return input
end

function getTrainData()
    store = readFileAndCreateDictorinary(FILE_PATH)
    word_dict = torch.load(DICTIONARY_FILE)
    train_data = makeTrainingData(store, WINDOW_SIZE, word_dict)
    return train_data
end

print(getTrainData())
--net, crit = construct_nn(WINDOW_SIZE, WORD_VEC_SIZE, 50)
--for idx, ins in ipairs(train_data) do
--   data = trainAndUpdatedWordVec(net, crit, ins.data, ins.label)
--    os.exit()
--end