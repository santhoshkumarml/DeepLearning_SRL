--
-- User: santhosh
-- Date: 11/30/15
--
local test_sent_start = 4413
local test_sent_end = 5413
local w2vutils = require 'w2vutils'
local UNK = torch.Tensor(WORD_VEC_SIZE):fill(0)
function test_SRL()
    local arg_ds = torch.load(ARGS_DICT_FILE)
    local arg_to_class_dict, class_to_arg_dict = arg_ds[1], arg_ds[2]
    local f = io.open(SRL_TRAIN_FILE)
    local accuracy, total_ins = 0, 0

    for sent_num = 1, test_sent_start - 1 do
        local predicate_idx, words, args = f:read(), f:read(), f:read()
    end

    for sent_num = test_sent_start, test_sent_end do
        local predicate_idx = tonumber(f:read())
        local words = string.split(f:read(), " ")
        local args = string.split(f:read(), " ")
        print('Processing the sentence', sent_num)
        for widx1 = 1, #words do
            local word_of_interest, current_arg = words[widx1], args[widx1]
            local feature_vecs_for_sent = torch.Tensor(#words + 2, WORD_VEC_SIZE
                    + SRL_WORD_INTEREST_DIST_DIM + SRL_VERB_DIST_DIM):fill(0)
            for widx2 = 1, #words do
                local curr_word = words[widx2]
                local feature_vec_for_word = w2vutils:word2vec(curr_word)
                if not feature_vec_for_word then
                    feature_vec_for_word = UNK
                    --print('Word Vec not known for', curr_word)
                else
                    feature_vec_for_word = feature_vec_for_word:narrow(1, 1, WORD_VEC_SIZE)
                end

                --Convert distance to binary tensor and append it to word vector
                local distance_to_word_of_interest = intToBin(widx1 - widx2)
                local distance_to_predicate = intToBin(predicate_idx - widx2)
                local feature_vec = torch.cat(
                    torch.cat(feature_vec_for_word, distance_to_word_of_interest),
                    distance_to_predicate)
                distance_to_predicate:free()
                distance_to_word_of_interest:free()
                feature_vecs_for_sent[widx2 + 1] = feature_vec
            end

            local real_target = arg_to_class_dict[current_arg]

            update_nn_for_sentence(feature_vecs_for_sent)

            local pred_target = global_net:forward(feature_vecs_for_sent)
            if real_target == pred_target then accuracy = accuracy +1 end
            total_ins = total_ins + 1

            feature_vecs_for_sent:free()
        end
    end
    accuracy = accuracy / total_ins
    return accuracy
end

function main()
    init_nn(true)
    local accuracy = test_SRL()
    print(accuracy)
end

main()