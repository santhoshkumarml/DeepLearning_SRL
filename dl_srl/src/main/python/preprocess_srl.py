import nltk

nltk.data.path.append('/media/santhosh/Data/workspace/nltk_data/')

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


def getSRLInfo(inst, sent, visited_dict = dict(), roles = set()):
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
        roles.add(arg)

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


if __name__ == '__main__':
    insts = nltk.corpus.propbank.instances()[:1]
    roles = set()
    for inst in insts:
        sent = nltk.corpus.treebank.sents(inst.fileid)[inst.sentnum]
        predicate_idx, sent_widx_to_arg_dict = getSRLInfo(inst, sent, roles=roles)
        print predicate_idx + 1
        print " ".join([sent[idx] for idx in range(len(sent))])
        print " ".join([sent_widx_to_arg_dict[idx] for idx in range(len(sent))])
