require 'torch';


START = "$START$"
FINISH = "$END$"
FILE_PATH = "dataset/language_model.data"

-- read and Store Data
function readFile(file_path)
	local f = io.open(file_path)
	store = {}
	while true do
	    local l = f:read()
	    if not l then break end
	    words = {}
	    table.insert(words, START)
	    for word in l:gmatch("%w+") do table.insert(words, word) end
	    table.insert(words,FINISH)
	    table.insert(store, words)
	end
	return store
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
   
store = readFile(FILE_PATH)
print(makeTrainingData(store, 11))