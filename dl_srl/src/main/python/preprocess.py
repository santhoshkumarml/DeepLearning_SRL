import nltk
from nltk import *
import re
import os
import random

nltk.data.path.append('/media/santhosh/Data/workspace/nltk_data')
rsrc_path = os.path.join('/home/santhosh/workspaces/DeepLearning_SRL/dl_srl', 'src/main/resources')
data_file_path = os.path.join(rsrc_path, 'wikipedia2text-extracted.txt')
op_data_file_path = os.path.join(rsrc_path, 'data.txt')
train_file_path = os.path.join(rsrc_path, 'train_data.txt')
diction_file_path = os.path.join(rsrc_path, 'diction.txt')
WINDOW_SIZE = 5
MID = WINDOW_SIZE/2

START = '$START$'
END = '$END$'
UNK = 'UNK'

def makeWindowAndTrainingData(diction):
    with open(op_data_file_path) as fp:
        with open(train_file_path, 'w') as tfp:
            for sent in fp:
                words = sent.split()
                words = [START for i in range(MID)] + words + [END for i in range(MID)]
                print len(words) - WINDOW_SIZE + 1
                for idx in range(0, len(words) - WINDOW_SIZE + 1):
                    pos_window_words = words[idx: idx + WINDOW_SIZE]

                    main_word = pos_window_words[MID]
                    new_word = None

                    while not new_word or new_word == main_word:
                        new_word = random.choice(tuple(diction))

                    neg_window_words = pos_window_words[:MID] + [new_word] + pos_window_words[MID+1:]

                    for word in pos_window_words:
                        tfp.write(word+" ")
                    tfp.write('\n')

                    for word in neg_window_words:
                        tfp.write(word+" ")
                    tfp.write('\n')


def tokenizeAndFormDict():
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    diction = set()
    with open(data_file_path) as f:
        filelines = f.read()
        filelines = filelines.replace('\r\n', ' ')
        filelines = filelines.replace('\n', ' ')
        sents = sent_detector.tokenize(filelines.decode('utf-8').strip())
        with open(op_data_file_path, 'w') as fp:
            for sent in sents:
                words =  nltk.word_tokenize(sent)
                for word in words:
                    fp.write(word.encode('utf-8')+" ")
                    diction.add(word)
                fp.write('\n')
    return diction

def saveDictionaryWords(diction):
    with open(diction_file_path, 'w') as fp:
        for word in diction:
            fp.write(word+"\n")

if __name__ == '__main__':
    diction = tokenizeAndFormDict()
    makeWindowAndTrainingData(diction)
    diction.add(START)
    diction.add(END)
    diction.add(END)
    diction.add(UNK)
    saveDictionaryWords(diction)