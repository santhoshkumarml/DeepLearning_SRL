import nltk
from nltk import *
import re
import os
import random
from collections import deque

nltk.data.path.append('/home/sriganesh/nltk_data')
rsrc_path = os.path.join('/home/sriganesh/Documents/CompLing/DeepLearning_SRL/dl_srl', 'src/main/resources')
data_file_path = os.path.join(rsrc_path, 'wikipedia2text-extracted.txt')
op_data_file_path = os.path.join(rsrc_path, 'data.txt')
train_file_path = os.path.join(rsrc_path, 'train_data.txt')
diction_file_path = os.path.join(rsrc_path, 'diction.txt')
WINDOW_SIZE = 5
MID = WINDOW_SIZE/2

START = '$START$'
END = '$END$'
UNK = 'UNK'

#number of lines processed locally = 128170:The Tang capital of Chang'an ( today 's Xi'an ) became an important center for Buddhist thought .

def makeWindowAndTrainingData(diction):
    with open(op_data_file_path) as fp:
        with open(train_file_path, 'w') as tfp:
            for sent in fp:
                words = sent.split()
                words = [START for i in range(MID)] + words + [END for i in range(MID)]
                pos_window_words = deque(words[0:WINDOW_SIZE+1])
                count = 0
                for idx in range(WINDOW_SIZE + 1, len(words)):
                    main_word = words[MID+count]
                    new_word = words[MID+count]
                    while new_word == main_word:
                        new_word = random.choice(tuple(diction))
                    tfp.write(' '.join(pos_window_words) + '\n')
                    pos_window_words[MID], temp = new_word, main_word
                    tfp.write(' '.join(pos_window_words) + '\n')
                    pos_window_words.popleft()
                    pos_window_words.append(words[idx])
                    count += 1


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
                words = nltk.word_tokenize(sent)
                for word in words:
                    fp.write(word.encode('utf-8')+" ")
                    diction.add(word.encode('utf-8'))
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
    diction.add(UNK)
    saveDictionaryWords(diction)