--
-- User: santhosh
-- Date: 10/29/15
--
require 'torch';
require 'nn';
require 'Constants';
require 'Heap';

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

--ts = m.removeDuplicates(dataset) -- remove duplicates from dataset

-- embeddings:
--p = m.embedding.random(t, {dim=2})  -- embed samples into a 2D plane, using random projections
--p = m.embedding.lle(t, {dim=2, neighbors=3})  -- embed samples into a 2D plane, using 3 neighbor (LLE)

--p = m.embedding.tsne(dataset, {dim=2, perplexity=30})  -- embed samples into a 2D plane, using tSNE
function findTopKNeighbors(word, k, hookWordVecLoader)
    local mlp = nn.CosineDistance()
    local h = Heap:new()
    local f = io.open(WORDS_FILE_PATH)
    local x = hookWordVecLoader(word)
    local knn = {}
    while true do
        local neigh_word = f:read()
        if not neigh_word then break end
        local y = hookWordVecLoader(neigh_word)
        if  y~= nil then
            local distance = mlp:forward({x, y})[1]
            if h:size() < k then
                h:push(neigh_word, distance)
            else
                local curr_word, min_dis = h:pop()
                if min_dis < distance then
                    h:push(neigh_word, distance)
                else
                    h:push(curr_word, min_dis)
                end
            end
        end
    end
    while not h:isempty() do
        local curr_word, min_dis = h:pop()
        table.insert(knn, curr_word)
    end
    return knn
end
function findKNNByGoogleWordVec(word, k)
    local w2vutils = require 'w2vutils'
    function googleWordVecHook(dict_word)
        local word_vec = w2vutils:word2vec(word)
        return word_vec
    end
    local knn = findTopKNeighbors(word, k, googleWordVecHook)
    w2vutils = nil
    collectgarbage()
    return knn
end


function findKNNAfterDomainAdaptation(word, k)
    local word_dict = torch.load(DICTIONARY_FILE)
    function languageModelWordVecHook(dict_word)
        local word_vec = word_dict[word]
        return word_vec
    end
    local knn = findTopKNeighbors(word, k, googleWordVecHook)
    return knn
end

local word = 'absorption'
local k = 5
print('Google Word Vec Neighbors', findKNNByGoogleWordVec(word, k))
print('LM Word Vec Neighbors', findKNNAfterDomainAdaptation(word, k))