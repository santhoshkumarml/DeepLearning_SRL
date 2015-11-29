import nltk
import Constants
import preprocess_srl

nltk.data.path.append(Constants.NLTK_DATA_PATH)
# conll_reader = nltk.corpus.reader.conll.ConllCorpusReader()
# conll_reader.srl_instances(root='conll2002',)

# conll = nltk.corpus.conll2007
# print conll.fileids()
#
# WORDS = 'words'   #: column type for words
# POS = 'pos'       #: column type for part-of-speech tags
# TREE = 'tree'     #: column type for parse trees
# CHUNK = 'chunk'   #: column type for chunk structures
# NE = 'ne'         #: column type for named entities
# SRL = 'srl'       #: column type for semantic role labels
# IGNORE = 'ignore' #: column type for column that should be ignored
#
# #: A list of all column types supported by the conll corpus reader.
# COLUMN_TYPES = (WORDS, POS, TREE, CHUNK, NE, SRL, IGNORE)
#
#
# path = nltk.data.find('corpora/conll2007')
# reader = nltk.corpus.reader.ConllChunkCorpusReader(path, conll.fileids(), COLUMN_TYPES)
# print reader.srl_instances()
# parser = nltk.parse.
# sents = conll.sents()[:2]
# for sent in sents:
#     parser.parse(sent)
def getSentences(insts):
    visited_sents = set()
    for inst in insts:
        if (inst.fileid, inst.sentnum) not in visited_sents:
            sent = nltk.corpus.treebank.sents(inst.fileid)[inst.sentnum]
            print ' '.join([s.encode('utf-8') for s in sent])
            visited_sents.add((inst.fileid, inst.sentnum))

def checkEqual(arr1, arr2):
    for idx in range(0, len(arr1)):
        if arr1[idx] != arr2[idx]:
            return False
    return True

def findSubarrayIdx(array, subarray):
    s_length = len(subarray)
    start_idx, end_idx = -1, -1
    for idx in range(s_length - 1, len(array)):
        spliced_array = array[idx - (s_length -1): idx + 1]
        if checkEqual(spliced_array, subarray):
            start_idx, end_idx = idx - (s_length -1), idx+1
    return start_idx, end_idx


if __name__ == '__main__':
  insts = nltk.corpus.propbank.instances()
  preprocess_srl.printSRLRoles(insts)
  # preprocess_srl.printSRLInfo(insts)

