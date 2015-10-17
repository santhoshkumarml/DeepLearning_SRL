require 'torch';


START = "$START$"
FINISH = "$END$"
FILE_PATH = "dataset/language_model.data"

-- read and Store Data
function readFile(file_path)
    local f = io.open(file_path)
    store = {}
    word_dict = {}
    while true do
        local l = f:read()
        if not l then break end
        words = {}
        table.insert(words, START)
        for word in l:gmatch("%w+") do 
            table.insert(words, word) 
            word_dict[word] = true
        end
        table.insert(words,FINISH)
        table.insert(store, words)
    end
    return store, word_dict
end

-- split into windows and make training data
function makeTrainingData(store, window_size)
    train_data = {}
    for idx, line in ipairs(store) do
        size = table.getn(line)
        no_of_windows = math.max( 1, size - window_size)

        for widx = 1, no_of_windows do

            training_inst = {}
            data = {}

            word_idx = widx+1

            while table.getn(data) < window_size and word_idx < size do
                table.insert(data, line[word_idx])
                word_idx = word_idx + 1
            end
            
            while word_idx >= size and table.getn(data) < window_size do
                table.insert(data, FINISH)
                word_idx = word_idx + 1
            end
            
            training_inst.data = data
            training_inst.label = '1'
            table.insert(train_data, training_inst)
        end   
    end
    return train_data
end
   
function construct_nn(windows_size, word_vec_size, hidden_layer_nodes)
    -- Add NN Layers
    net = nn.Sequential()
    net:add(nn.Linear(window_size * word_vec_size, hidden_layer_nodes))
    net:add(nn.Sigmoid())
    net:add(nn.Linear(hidden_layer_nodes, 1))
    net:add(nn.LogSoftMax())
    -- Define Loss Function
    criterion = nn.ClassNLLCriterion()
    return net, criterion
end

function trainAndUpdatedWordVec(net, criterion, input, output)
    for i = 1, 20 do
        -- feed it to the neural network and the criterion
        criterion:forward(net:forward(input), real_output)

        -- train over this example in 3 steps
        -- (1) zero the accumulation of the gradients
        net:zeroGradParameters()

        -- (2) accumulate gradients
        net:backward(input, criterion:backward(net.output, output))
        
        grad_ip = net.gradInput
        
        print(grad_ip)
                
        -- (3) update parameters with a 0.01 learning rate
        net:updateParameters(0.01)
    end
    return data
end