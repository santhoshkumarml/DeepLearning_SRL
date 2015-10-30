--
-- Created by IntelliJ IDEA.
-- User: santhosh
-- Date: 10/29/15
-- Time: 7:39 PM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
m = require 'manifold'

WORDS_FILE_PATH = "../resources/diction.txt"
DICTIONARY_FILE = "../resources/dictionary.dict"
WORD_VEC_SIZE = 25

function formData()
    --dictionary
    local f = io.open(WORDS_FILE_PATH)

    -- a dataset:
    local word_dict = torch.load(DICTIONARY_FILE)

    local dataset = nil
    local widx = 1
    local words = {}
    while true do
        local l = f:read()
        if not l then break end
        words[widx] = l
        widx = widx + 1
        word_vec = word_dict[l]:reshape(1, WORD_VEC_SIZE)
        if dataset == nil then
            dataset = word_vec
        else
            dataset = torch.cat(dataset, word_vec, 1)
        end
    end
    return dataset, words
end


-- basic functions:
--ns = m.neighbors(t) -- return the matrix of neighbors for all samples (sorted)
--ds = m.distances(t) -- return the matrix of distances (L2)
--ts = m.removeDuplicates(dataset) -- remove duplicates from dataset

-- embeddings:
--p = m.embedding.random(t, {dim=2})  -- embed samples into a 2D plane, using random projections
--p = m.embedding.lle(t, {dim=2, neighbors=3})  -- embed samples into a 2D plane, using 3 neighbor (LLE)
local dataset, words = formData()
p = m.embedding.tsne(dataset, {dim=2, perplexity=30})  -- embed samples into a 2D plane, using tSNE

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

