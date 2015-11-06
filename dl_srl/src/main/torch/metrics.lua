--
-- User: santhosh
-- Date: 10/29/15
--
require 'torch';
require 'nn';
require 'Constants';
require 'Heap'
m = require 'manifold'

function formData()
    --dictionary
    local f = io.open(WORDS_FILE_PATH)

    -- a dataset:
    local word_dict = torch.load(DICTIONARY_FILE)

    local dataset = nil
    local widx = 1
    local idx_words = {}
    local words_idx = {}
    while true do
        local l = f:read()
        if not l then break end
        if word_dict[l] ~= nil then
            idx_words[widx] = l
            --words_idx[l] = widx
            widx = widx + 1
            word_vec = word_dict[l]:reshape(1, WORD_VEC_SIZE)
            if dataset == nil then
                dataset = word_vec
            else
                dataset = torch.cat(dataset, word_vec, 1)
            end
            word_dict[l] = nil
        end
    end
    return dataset, idx_words
end

function plotWord2Vec(p)
    Plot = require 'itorch.Plot'
    -- scatter plots
    plot = Plot()
    for i =1, p:size(1) do
        x1 = torch.Tensor(1)
        y1 = torch.Tensor(1)
        x1[1] = p[i][1]
        y1[1] = p[i][2]
        plot:circle(x1, y1, 'red',words[i])
    end

    plot:title('Scatter Plot Demo'):redraw()
    plot:xaxis('Dim1'):yaxis('Dim2'):redraw()
    plot:redraw()
    plot:legend(true)
    -- print(plot:toHTML())
    plot:save('out.html')

end

function findTopKNeighbors(k)
    local word_dict = torch.load(DICTIONARY_FILE)
    local f = io.open(WORDS_FILE_PATH)
    mlp = nn.CosineDistance()
    x = word_dict['when']
    h = Heap:new()
    while true do
        local l = f:read()
        if not l then break end
        if l ~= 'when' and word_dict[l] ~= nil then
            distance = mlp:forward({x, word_dict[l]})[1]
            if h:size() < k then
                h:push(l, distance)
            else
                word, min_dis = h:pop()
                if min_dis < distance then
                    h:push(l, distance)
                else
                    h:push(word, min_dis)
                end
            end
        end
    end
    while not h:isempty() do print(h:pop()) end
end

--ds = m.distances(dataset) -- return the matrix of distances (L2)
local dataset, idx_words = formData()
ns = m.neighbors(dataset) -- return the matrix of neighbors for all samples (sorted)

for i = 1, 143 do
    word = idx_words[i]
    if word == 'when' then
        print(word)
        for j = 1, 10 do
            idx = ns[i][j]
            print(idx, idx_words[idx])
        end
    end
end
--ts = m.removeDuplicates(dataset) -- remove duplicates from dataset

-- embeddings:
--p = m.embedding.random(t, {dim=2})  -- embed samples into a 2D plane, using random projections
--p = m.embedding.lle(t, {dim=2, neighbors=3})  -- embed samples into a 2D plane, using 3 neighbor (LLE)

--p = m.embedding.tsne(dataset, {dim=2, perplexity=30})  -- embed samples into a 2D plane, using tSNE

