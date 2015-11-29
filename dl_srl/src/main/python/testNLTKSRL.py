import nltk
import preprocess
import os
import preprocess_srl

nltk.data.path.append('/media/santhosh/Data/workspace/nltk_data/')
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

ARGS_DICT_FILE_NAME = 'args.txt'
SRL_TRAIN_FILE_NAME = 'SRL_train.txt'
SRL_TRAIN_FILE = os.path.join(preprocess.rsrc_path, SRL_TRAIN_FILE_NAME)
ARGS_DICT_FILE = os.path.join(preprocess.rsrc_path, ARGS_DICT_FILE_NAME)

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


def getTreeLeafPos(tpos, tree):
    all_leaf_pos = []
    stack = list()
    visited_pos = set()
    stack.append((tpos, tree[tpos]))
    while len(stack) > 0:
        i, node = stack.pop()
        visited_pos.add(i)
        if isinstance(node, nltk.tree.Tree):
            childpos = [tuple(list(i)+list(p)) for p in node.treepositions()
                        if tuple(list(i)+list(p)) not in visited_pos]
            for pos in childpos:
                stack.append((pos, tree[pos]))
        else:
            all_leaf_pos.append(i)
    return all_leaf_pos

if __name__ == '__main__':
    insts = nltk.corpus.propbank.instances()[112913:112914]
    for inst in insts:
        sent = nltk.corpus.treebank.sents(inst.fileid)[inst.sentnum]
        print sent
        tree = inst.tree
        all_leaves_positions = {tree.leaf_treeposition(i): i for i in range(len(tree.leaves()))}
        for locArg in inst.arguments:
            print '---------------------------------------------------------'
            loc, arg = locArg
            for propBankPtr in preprocess_srl.getPropBankTreePointers(loc):
                tpos = propBankPtr.treepos(tree)
                leaf_positions = getTreeLeafPos(tpos, tree)
                sent_word_idxs = sorted([all_leaves_positions[leaf_pos] for leaf_pos in leaf_positions])
                print [sent[idx] for idx in sent_word_idxs]
            print '---------------------------------------------------------'

