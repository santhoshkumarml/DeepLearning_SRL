import nltk
import os
import Constants

nltk.data.path.append(Constants.NLTK_DATA_PATH)

def getCoarseGrainedArg(arg):
    pass

def getPropBankTreePointers(loc):
    if isinstance(loc, nltk.corpus.reader.propbank.PropbankTreePointer):
        return [loc]
    elif isinstance(loc, nltk.corpus.reader.propbank.PropbankChainTreePointer) or \
            isinstance(loc, nltk.corpus.reader.propbank.PropbankSplitTreePointer):
        locs = loc.pieces
        all_locs = []
        for loc in locs:
            all_locs.extend(getPropBankTreePointers(loc))
        return all_locs


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


def getSRLInfo(inst, sent):
    tree = inst.tree
    sent_widx_to_arg_dict = dict()

    all_leaves_positions = {tree.leaf_treeposition(i): i for i in range(len(tree.leaves()))}

    for locArg in inst.arguments:
        loc, arg = locArg
        for propBankPtr in getPropBankTreePointers(loc):
            tpos = propBankPtr.treepos(tree)
            leaf_positions = getTreeLeafPos(tpos, tree)
            sent_word_idxs = sorted([all_leaves_positions[leaf_pos] for leaf_pos in leaf_positions])
            for idx in sent_word_idxs:
                sent_widx_to_arg_dict[idx] = getCoarseGrainedArg(arg)

    #Predicate
    loc = inst.predicate
    predicate_idx = -1
    for propBankPtr in getPropBankTreePointers(loc):
        tpos = propBankPtr.treepos(tree)
        leaf_positions = getTreeLeafPos(tpos, tree)
        sent_word_idxs = sorted([all_leaves_positions[leaf_pos] for leaf_pos in leaf_positions])
        assert (len(sent_word_idxs) == 1)
        predicate_idx = sent_word_idxs[0]
        for idx in sent_word_idxs:
            sent_widx_to_arg_dict[idx] = 'VERB'

    for idx in range(0, len(sent)):
        if idx not in sent_widx_to_arg_dict:
            sent_widx_to_arg_dict[idx] = 'NULL'

    # for key in sorted(sent_widx_to_arg_dict.keys()):
    #     print key, sent_widx_to_arg_dict[key]

    return predicate_idx, sent_widx_to_arg_dict

def printSRLRoles(insts):
    all_roles = set([arg for inst in insts for loc, arg in inst.arguments])
    roles = set([getCoarseGrainedArg(role) for role in all_roles])
    output = ','.join([role for role in roles])
    with open(Constants.ARGS_DICT_FILE, 'w') as f:
        f.write(output)

def printSRLInfo(insts):
    with open(Constants.SRL_TRAIN_FILE, 'w') as f:
        for inst in insts:
            sent = nltk.corpus.treebank.sents(inst.fileid)[inst.sentnum]
            predicate_idx, sent_widx_to_arg_dict = getSRLInfo(inst, sent)
            f.write(str(predicate_idx + 1) + '\n')
            output1 = ' '.join([sent[idx] for idx in range(len(sent))])
            f.write(output1 + '\n')
            output2 = ' '.join([sent_widx_to_arg_dict[idx] for idx in range(len(sent))])
            f.write(output2 + '\n')