import nltk
from nltk import *
import re
from nltk.tag.stanford import POSTagger

nltk.data.path.append('/media/santhosh/Data/workspace/nltk_data')

with open(file_path) as f:
    filelines = f.read()
    filelines = filelines.replace('\r\n', ' ')
    filelines = filelines.replace('\n', ' ')
    sents = sent_detector.tokenize(filelines.decode('utf-8').stip())
    # print type(filelines), len(filelines), filelines, genre_file_path
    # filelines = filelines[0]
    # print filelines
    sents = sents[:1000]