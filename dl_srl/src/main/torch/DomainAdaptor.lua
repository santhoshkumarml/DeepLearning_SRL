--
-- User: santhosh
-- Date: 12/5/15
--

require 'Constants'
require 'torch'
require 'nn'
require 'MyStochasticGradient'
torch.setdefaulttensortype('torch.FloatTensor')
local w2vutils = nil

local ksz = 3
local convOutputFrame = 100
local HUs = 100
--Later loaded to number of argument classes
local old_domain_final_output_layer_size = -1
local new_domain_final_output_layer_size = -1
local UNK = torch.Tensor(WORD_VEC_SIZE):fill(0)

local domain_global_net = {}
local trainer = nil

function init_nn(isLoadNewDomainNet)
  local old_f = io.open(SRL_TEMPORAL_NET_FILE)
  if not old_f then error('Old Domain SRL Net should be present') end
  local new_f = io.open(NEW_DOMAIN_SRL_TEMPORAL_NET_FILE)
  if isLoadNewDomainNet and new_f ~= nil then
    domain_global_net = torch.load(NEW_DOMAIN_SRL_TEMPORAL_NET_FILE)
    if new_f~=nil then new_f:close() end
    print('New Domain Net Loaded')
  else
    domain_global_net = torch.load(SRL_TEMPORAL_NET_FILE)
    --Add a new linear layer after exisisting nn.Linear
    domain_global_net:insert(nn.Tanh(), 5)
    domain_global_net:insert(nn.Linear(old_domain_final_output_layer_size,
      new_domain_final_output_layer_size), 6)
    domain_global_net:insert(nn.Tanh(), 7)
    print(tostring(domain_global_net))
  end
  old_f:close()

  local criterion = nn.ClassNLLCriterion()
  trainer = nn.StochasticGradient(domain_global_net, criterion)
  trainer.verbose = false
  trainer.maxIteration = 1
  trainer.learningRate = 0.0001
end

function update_nn_for_sentence(sentence)
  local sentence_size = sentence:size(1)
  sentence_size = sentence_size - (2 * math.floor(ksz/2))
  domain_global_net.modules[2] = nn.TemporalMaxPooling(sentence_size)
end

--Serialize the neural net
function save_nn()
  torch.save(NEW_DOMAIN_SRL_TEMPORAL_NET_FILE, domain_global_net)
end


--Encode the number in 5 dimenion space
--Also the number is restrained to stick to [-15, 15] if it is not
function intToBin(num)
  local isNeg = false
  if num > 15 then num = 15 else if num < -15 then num = -15 end end
  if num < 0 then isNeg = true end
  local num = math.abs(num)
  local tensor = torch.Tensor(5):fill(0)
  local start_idx = 5
  while num > 0 do
    local div = (num % 2)
    num = math.floor(num / 2)
    tensor[start_idx] = div
    start_idx = start_idx - 1
  end
  if isNeg then tensor[1] = 1 else tensor[1] = 0 end
  return tensor
end
function makeArgToClassDict(argsFile, argsDictFile)
  local f = io.open(argsFile)
  local args = string.split(f:read(), ",")
  local arg_ds = {}
  local arg_to_class_dict, class_to_arg_dict = {}, {}
  for idx = 1, #args do
    class_to_arg_dict[idx] = args[idx]
    arg_to_class_dict[args[idx]] = idx
  end
  arg_ds[1] = arg_to_class_dict
  arg_ds[2] = class_to_arg_dict
  torch.save(argsDictFile, {arg_to_class_dict, class_to_arg_dict})
  f:close()
  return #args
end

--Read Old Domain Arguments dictionary and generate class number for them starting with 1
function makeArgToClassDictOldDomain()
  return makeArgToClassDict(ARGS_FILE, ARGS_DICT_FILE)
end

--Read New Domain Arguments dictionary and generate class number for them starting with 1
function makeArgToClassDictNewDomain()
  return makeArgToClassDict(NEW_DOMAIN_ARGS_FILE, NEW_DOMAIN_ARGS_DICT_FILE)
end

--Train on an instance of sentences with a specific word of interest.
function trainForSingleInstance(train_data)
  local sentence = train_data[1][1]
  local target = train_data[1][2]
  update_nn_for_sentence(sentence)
  trainer:train(train_data)
end


--Find the class number from the LogSoftMax final layer
function findClasNumFromProbs(probs)
  local probs = probs[1]
  local size = probs:size(1)
  local max_prob, max_index = -4000, -1
  for index = 1, size do
    local curr_prob = probs[index]
    if curr_prob > max_prob then
      max_prob = curr_prob
      max_index = index
    end
  end
  return max_index
end

--Create And Get Confusion Matrix
function createConfusionMatrix()
  local arg_ds = torch.load(NEW_DOMAIN_ARGS_DICT_FILE)
  local arg_to_class_dict, class_to_arg_dict = arg_ds[1], arg_ds[2]
  local confusion_matrix = {}
  for cl1, arg1 in ipairs(class_to_arg_dict) do
    confusion_matrix[arg1] = {}
    for cl2, arg2 in ipairs(class_to_arg_dict) do
      confusion_matrix[arg1][arg2] = 0.0
    end
  end
  return confusion_matrix
end

function updateConfusionMatrix(confusion_matrix, real_arg, predicted_arg)
  confusion_matrix[real_arg][predicted_arg] = confusion_matrix[real_arg][predicted_arg] + 1.0
end

function printConfusionMatrix(confusion_matrix)
  local arg_ds = torch.load(NEW_DOMAIN_ARGS_DICT_FILE)
  local arg_to_class_dict, class_to_arg_dict = arg_ds[1], arg_ds[2]
  io.write(" ")
  for cl1, arg1 in ipairs(class_to_arg_dict) do
    io.write("\t", arg1)
  end
  io.write("\n")
  for cl1, arg1 in ipairs(class_to_arg_dict) do
    io.write(arg1, "\t")
    for cl2, arg2 in ipairs(class_to_arg_dict) do
      io.write(confusion_matrix[arg1][arg2], "\t")
    end
    io.write("\n")
  end
end

--function calculatePrecisionRecallF1(confusion_matrix)
--  local arg_ds = torch.load(NEW_DOMAIN_ARGS_DICT_FILE)
--  local arg_to_class_dict, class_to_arg_dict = arg_ds[1], arg_ds[2]
--  local precision, recall = {}, {}
--  for cl1, arg1 in ipairs(class_to_arg_dict) do
--    local tp = confusion_matrix[arg1][arg1]
--    local tpfp = 0.0
--    for cl2, arg2 in ipairs(class_to_arg_dict) do
--      tpfp = tpfp + confusion_matrix[arg2][arg1]
--    end
--    precision[arg1] = tp / tpfp
--  end
--  for cl1, arg1 in ipairs(class_to_arg_dict) do
--    local tp = confusion_matrix[arg1][arg1]
--    local tpfn = 0.0
--    for cl2, arg2 in ipairs(class_to_arg_dict) do
--      tpfn = tpfn + confusion_matrix[arg1][arg2]
--    end
--    recall[arg1] = tp / tpfn
--  end
--  for cl, arg in ipairs(class_to_arg_dict) do
--    print('Class:', arg, 'Precision:', precision[arg], 'Recall:', recall[arg],
--    'F1:', (2 * precision[arg] * recall[arg]) / (precision[arg] + recall[arg]))
--  end
--end

--Construct Feature Vector instance
function constructFeatureVecForSentence(predicate_idx, word_of_interest_idx, words)
  local feature_vecs_for_sent = torch.Tensor(#words + 2, WORD_VEC_SIZE
    + SRL_WORD_INTEREST_DIST_DIM + SRL_VERB_DIST_DIM):fill(0)
  for widx = 1, #words do
    local curr_word = words[widx]
    local feature_vec_for_word = w2vutils:word2vec(curr_word)
    if not feature_vec_for_word then
      feature_vec_for_word = UNK
      --print('Word Vec not known for', curr_word)
    else
      feature_vec_for_word = feature_vec_for_word:narrow(1, 1, WORD_VEC_SIZE)
    end

    --Convert distance to binary tensor and append it to word vector
    local distance_to_word_of_interest = intToBin(word_of_interest_idx - widx)
    local distance_to_predicate = intToBin(predicate_idx - widx)
    local feature_vec = torch.cat(
      torch.cat(feature_vec_for_word, distance_to_word_of_interest),
      distance_to_predicate)
    distance_to_predicate:free()
    distance_to_word_of_interest:free()
    feature_vecs_for_sent[widx + 1] = feature_vec
  end
  return feature_vecs_for_sent
end

--Train for sentences
function train(epoch, epoch_checkpt, sent_checkpt, train_sent_start, train_sent_end)
  if train_sent_end == -1 then return -1 end

  print('--------------------------Train iteration number:'..epoch..'----------------------------------------')
  -- load data structures for class_to_arg_name conversion and arg_name_to_class conversion
  local arg_ds = torch.load(NEW_DOMAIN_ARGS_DICT_FILE)
  local arg_to_class_dict, class_to_arg_dict = arg_ds[1], arg_ds[2]
  local f = io.open(NEW_DOMAIN_SRL_TRAIN_FILE)
  local checkpt_ctr = 0

  for sent_num = 1, train_sent_start - 1 do
    local predicate_idx, words, args = f:read(), f:read(), f:read()
  end
  local count = 0
  for sent_num = train_sent_start, train_sent_end do
    local predicate_idx = tonumber(f:read())
    local wordString = f:read()
    local argString = f:read()
    local words = string.split(wordString, " ")
    local args = string.split(argString, " ")
    local count = 0
    if epoch > epoch_checkpt or sent_num > sent_checkpt then
      collectgarbage()
      print('Processing the sentence', sent_num)
      for word_of_interest_idx = 1, #words do
        local word_of_interest, current_arg = words[word_of_interest_idx], args[word_of_interest_idx]
        local curr_target = arg_to_class_dict[current_arg]
        if((count < 1 and curr_target == 5) or curr_target < 5) then
          local feature_vecs_for_sent = constructFeatureVecForSentence(predicate_idx,
            word_of_interest_idx, words)

          local curr_target = arg_to_class_dict[current_arg]
          local train_data = {}
          train_data[1] = {feature_vecs_for_sent, curr_target }
          function train_data:size() return 1 end

          trainForSingleInstance(train_data)
          feature_vecs_for_sent:free()
          if(curr_target == 5) then
            count = count +1
          end
        end
      end
      if checkpt_ctr % 25 == 0 then
        save_nn()
        torch.save(NEW_DOMAIN_SRL_CHECKPT_FILE, {epoch, sent_num})
      end
      checkpt_ctr = checkpt_ctr + 1
    else
      print('Skipped Processing Sentence:', sent_num, 'Epoch:', epoch)
    end
  end
  save_nn()
  f:close()
  print('------------------------------------------------------------------------------------------')
end


--Run Test Set
function test_SRL(test_sent_start, test_sent_end)
  local arg_ds = torch.load(NEW_DOMAIN_ARGS_DICT_FILE)
  local arg_to_class_dict, class_to_arg_dict = arg_ds[1], arg_ds[2]
  local f = io.open(NEW_DOMAIN_SRL_TRAIN_FILE)
  if test_sent_end == -1 then return -1 end
  local confusion_matrix = createConfusionMatrix()
  for sent_num = 1, test_sent_start - 1 do
    local predicate_idx, words, args = f:read(), f:read(), f:read()
  end

  for sent_num = test_sent_start, test_sent_end do
    print('Processing the sentence', sent_num)
    local predicate_idx = tonumber(f:read())
    local words = string.split(f:read(), " ")
    local args = string.split(f:read(), " ")
    for word_of_interest_idx = 1, #words do
      local word_of_interest, current_arg = words[word_of_interest_idx], args[word_of_interest_idx]
      local feature_vecs_for_sent = constructFeatureVecForSentence(predicate_idx,
        word_of_interest_idx, words)

      local real_target = arg_to_class_dict[current_arg]
      update_nn_for_sentence(feature_vecs_for_sent)

      local logProbs = domain_global_net:forward(feature_vecs_for_sent)
      local probs = torch.exp(logProbs)
      local pred_target = findClasNumFromProbs(probs)

      updateConfusionMatrix(confusion_matrix, current_arg, class_to_arg_dict[pred_target])
      feature_vecs_for_sent:free()
    end
  end
  return confusion_matrix
end

function loadCheckPt()
  local checkPt = {1, 0}
  local f = io.open(NEW_DOMAIN_SRL_CHECKPT_FILE)
  if f ~= nil then
    checkPt = torch.load(NEW_DOMAIN_SRL_CHECKPT_FILE)
    f:close()
  end
  return checkPt
end


-- Remove the serialized nets
function doCleanup()
  os.remove(NEW_DOMAIN_SRL_TEMPORAL_NET_FILE)
  os.remove(NEW_DOMAIN_SRL_CHECKPT_FILE)
end

--Main Function
function domain_adapt(isTrain, ins_start, ins_end)
  --Number of different argument classes
  old_domain_final_output_layer_size = makeArgToClassDictOldDomain()
  new_domain_final_output_layer_size = makeArgToClassDictNewDomain()
  --print(old_domain_final_output_layer_size, new_domain_final_output_layer_size)
  init_nn(true)

  local checkPt = loadCheckPt()
  local epoch_checkpt, sent_checkpt = checkPt[1], checkPt[2]
  print('Epoch Check Point', epoch_checkpt)
  w2vutils = require 'w2vutils'

  if isTrain then
    for epoch = epoch_checkpt, EPOCH do
      train(epoch, epoch_checkpt, sent_checkpt, ins_start, ins_end)
    end
  else
    local confusion_matrix = test_SRL(ins_start, ins_end)
    printConfusionMatrix(confusion_matrix)
  end
end

if (#arg < 3) then
  error("Not Enough Arguments, Usage: th DomainAdaptor.lua \"train/test\" sent_start sent_end [clean]")
end

local ins_start, ins_end = tonumber(arg[2]), tonumber(arg[3])

local isCleanExistingNet = false

if #arg == 4 and arg[4] == 'clean' then
  isCleanExistingNet = true
  doCleanup()
end

if arg[1] == "train" then
  domain_adapt(true, ins_start, ins_end)
else
  domain_adapt(false, ins_start, ins_end)
end
