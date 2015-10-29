--
-- Created by IntelliJ IDEA.
-- User: santhosh
-- Date: 10/29/15
-- Time: 7:39 PM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
m = require 'manifold'

-- a dataset:
word_dict = torch.load(DICTIONARY_FILE)



-- basic functions:
ns = m.neighbors(t) -- return the matrix of neighbors for all samples (sorted)
ds = m.distances(t) -- return the matrix of distances (L2)
ts = m.removeDuplicates(t) -- remove duplicates from dataset

-- embeddings:
p = m.embedding.random(t, {dim=2})  -- embed samples into a 2D plane, using random projections
p = m.embedding.lle(t, {dim=2, neighbors=3})  -- embed samples into a 2D plane, using 3 neighbor (LLE)
p = m.embedding.tsne(t, {dim=2, perplexity=30})  -- embed samples into a 2D plane, using tSNE

