--
-- User: santhosh
-- Date: 11/3/15
--


WORDS_FILE_PATH = "../resources/diction.txt"

TRAIN_DATA_FILE_PATH = "../resources/train_data.txt"

DICTIONARY_FILE = "../resources/dictionary.dict"

LANGUAGE_NET_FILE = "../resources/language_model.net"

SRL_TEMPORAL_NET_FILE = "../resources/srl_temporal.net"
SRL_CHECKPT_FILE = "../resources/srl_checpt.t7"

NEW_DOMAIN_SRL_TEMPORAL_NET_FILE = "../resources/srl_temporal_new_domain.net"
NEW_DOMAIN_SRL_CHECKPT_FILE = "../resources/srl_checpt_new_domain.t7"

GOOGLE_PRETRAINED_WORD2_VEC_FILE = '/media/santhosh/Data/workspace/nlp_project/'..
        'SRL/word2vec/GoogleNews-vectors-negative300.bin'

GOOGLE_WORD2VEC_OUTPUT_FILE_NAME = '/media/santhosh/Data/workspace/nlp_project/SRL/word2vec/word2vec.t7'

WORD2VEC = {
    binfilename = GOOGLE_PRETRAINED_WORD2_VEC_FILE,
    outfilename = GOOGLE_WORD2VEC_OUTPUT_FILE_NAME
}

WORD_VEC_SIZE = 50

SRL_WORD_INTEREST_DIST_DIM = 5

SRL_VERB_DIST_DIM = 5

WINDOW_SIZE = 11

--will take BATCH_SIZE +ve and BATCH_SIZE -ve samples
BATCH_SIZE = 5

EPOCH = 2000

START = "$START$"

FINISH = "$END$"

ARGS_FILE = '../resources/args.txt'
ARGS_DICT_FILE = '../resources/args.t7'
NEW_DOMAIN_ARGS_FILE = '../resources/args_new_domain.txt'
NEW_DOMAIN_ARGS_DICT_FILE = '../resources/args_new_domain.t7'

SRL_TRAIN_FILE = '../resources/SRL_train.txt'

NEW_DOMAIN_SRL_TRAIN_FILE = '../resources/SRL_train_new_domain.txt'
