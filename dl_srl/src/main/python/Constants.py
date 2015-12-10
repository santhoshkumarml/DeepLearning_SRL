import os
import nltk

NLTK_DATA_PATH = '/home/smanavas/data/nltk_data/'
rsrc_path = os.path.join('/home/smanavas/ws/DeepLearning_SRL/dl_srl', 'src/main/resources')
data_file_path = os.path.join(rsrc_path, 'small_wikipedia2text-extracted.txt')
op_data_file_path = os.path.join(rsrc_path, 'data.txt')
train_file_path = os.path.join(rsrc_path, 'train_data.txt')
diction_file_path = os.path.join(rsrc_path, 'diction.txt')

ARGS_DICT_FILE_NAME = 'args.txt'
SRL_TRAIN_FILE_NAME = 'SRL_train.txt'
SRL_TRAIN_FILE = os.path.join(rsrc_path, SRL_TRAIN_FILE_NAME)
ARGS_DICT_FILE = os.path.join(rsrc_path, ARGS_DICT_FILE_NAME)



WINDOW_SIZE = 11
MID = WINDOW_SIZE/2
START = '$START$'
END = '$END$'
UNK = 'UNK'
