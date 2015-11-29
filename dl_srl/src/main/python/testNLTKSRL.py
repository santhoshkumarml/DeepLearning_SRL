import nltk
import preprocess
import os

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


def getPropBankTreePointers(loc):
    if isinstance(loc, nltk.corpus.reader.propbank.PropbankTreePointer):
        return [loc]
    elif isinstance(loc, nltk.corpus.reader.propbank.PropbankChainTreePointer) or\
            isinstance(loc, nltk.corpus.reader.propbank.PropbankSplitTreePointer):
        locs = loc.pieces
        all_locs = []
        for loc in locs:
            all_locs.extend(getPropBankTreePointers(loc))
        return all_locs

def getTreePosLeaves(loc, tree):
    treepos = loc.treepos(tree)
    return treepos, [tree.leaf_treeposition()]


def getSRLInfo(insts, idx, visited_dict = dict(), roles = set()):
    # print nltk.corpus.treebank.tagged_sents(inst.fileid)[inst.sentnum]
    # print inst.wordnum
    inst = insts[idx]
    sent_key = (inst.fileid, inst.sentnum)
    tree = inst.tree
    sent_widx_to_arg_dict = dict()
    # if sent_key not in visited_dict:
    #     visited_dict[sent_key] = 0
    # visited_dict[sent_key] = visited_dict[sent_key] + 1
    sent_array = nltk.corpus.treebank.sents(inst.fileid)[inst.sentnum]
    inst.tree.pretty_print(unicodelines=True, nodedist=4)
    # sent_subarray = loc.select(tree).leaves()
    # s, e = findSubarrayIdx(sent_array, sent_subarray)
    # print s, e, sent_subarray, arg
    print '------------------------------INST', idx, '-', sent_key, '-----------------------------------------'
    for arg in inst.arguments:
        loc, arg = arg
        for pos in getPropBankTreePointers(loc):
            sent_subarray = getTreePosLeaves(pos, tree)
            print sent_subarray
            # # s, e = findSubarrayIdx(sent_array, sent_subarray)
            # for i in range(s, e):
            #     sent_widx_to_arg_dict[i] = arg
        roles.add(arg)

    loc = inst.predicate
    for pos in getPropBankTreePointers(loc):
        sent_subarray = getTreePosLeaves(pos, tree)
        print sent_subarray
        # s, e = findSubarrayIdx(sent_array, sent_subarray)
        # for i in range(s, e):
        #     sent_widx_to_arg_dict[i] = 'PREDICATE'

    # for i in range(0, len(sent_array)):
    #     if i not in sent_widx_to_arg_dict:
    #         sent_widx_to_arg_dict[i] = 'NULL'

    for key in sorted(sent_widx_to_arg_dict.keys()):
        print key, sent_widx_to_arg_dict[key]

    print '-----------------------------------------------------------------------'

    return sent_key


if __name__ == '__main__':
    insts = nltk.corpus.propbank.instances()[112913:112914]
    visited_dict = dict()
    roles = set()
    for i in range(len(insts)):
        fileid, sentnum = getSRLInfo(insts, i, visited_dict, roles)
    #
    # print roles, len(roles)
    # for key in sorted(visited_dict.keys(), key=lambda key: visited_dict[key], reverse=True):
    #     print key, visited_dict[key]