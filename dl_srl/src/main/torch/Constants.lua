--
-- User: santhosh
-- Date: 11/3/15
--


WORDS_FILE_PATH = "../resources/diction.txt"

TRAIN_DATA_FILE_PATH = "../resources/train_data.txt"

DICTIONARY_FILE = "../resources/dictionary.dict"

LANGUAGE_NET_FILE = "../resources/language_model.net"

SRL_TEMPORAL_NET_FILE = "../resources/srl_temporal.net"
SRL_CONSTANT_NET_FILE = "../resources/srl_constant.net"

GOOGLE_PRETRAINED_WORD2_VEC_FILE = '/media/santhosh/Data/workspace/nlp_project/'..
        'SRL/word2vec/GoogleNews-vectors-negative300.bin'

GOOGLE_WORD2VEC_OUTPUT_FILE_NAME = '/media/santhosh/Data/workspace/nlp_project/SRL/word2vec/word2vec.t7'

WORD2VEC = {
    binfilename = GOOGLE_PRETRAINED_WORD2_VEC_FILE,
    outfilename = GOOGLE_WORD2VEC_OUTPUT_FILE_NAME
}

WORD_VEC_SIZE = 50

SRL_OTHER_DIMENSIONS = 2

WINDOW_SIZE = 11

--will take BATCH_SIZE +ve and BATCH_SIZE -ve samples
BATCH_SIZE = 5

EPOCH = 2000

START = "$START$"

FINISH = "$END$"

