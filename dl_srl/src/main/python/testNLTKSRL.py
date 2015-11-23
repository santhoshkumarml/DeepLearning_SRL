import nltk
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

def getSentences(insts):
    visited_sents = set()
    for inst in insts:
        if (inst.fileid, inst.sentnum) not in visited_sents:
            sent = nltk.corpus.treebank.sents(inst.fileid)[inst.sentnum]
            print ' '.join([s.encode('utf-8') for s in sent])
            visited_sents.add((inst.fileid, inst.sentnum))

def getSRLInfo(inst, visited_dict = dict()):
    # print nltk.corpus.treebank.tagged_sents(inst.fileid)[inst.sentnum]
    # print inst.wordnum
    key = (inst.fileid, inst.sentnum)
    if key not in visited_dict:
        visited_dict[key] = 0
    visited_dict[key] = visited_dict[key] + 1
    # print inst.predicate
    # print inst.roleset
    # print inst.arguments

if __name__ == '__main__':
    insts = nltk.corpus.propbank.instances()
    # inst = insts[11]
    visited_dict = dict()
    for inst in insts:
        getSRLInfo(inst, visited_dict)
    for key in sorted(visited_dict.keys(), key=lambda key: visited_dict[key], reverse=True):
        print key, visited_dict[key]