import pdb


from tree import Token, Tree, get_yield
import pickle

ID,FORM,LEMMA,CPOS,FPOS,MORPH,HEAD,DEPREL,PHEAD,PDEPREL=range(10)

def is_xml(s) : return s[0] == "<" and s[-1] == ">"
def is_xml_beg(s) : return is_xml(s) and s[1] != "/"
def is_xml_end(s) : return is_xml(s) and not is_xml_beg(s)
def is_head(s) : return is_xml(s) and "^head" in s


def get_nt_from_xml(s) :
    if is_xml_beg(s) :
        s = s[1:-1]
    elif is_xml_end(s) :
        s = s[2:-1]
    else : assert(False)
    if s[-5:] == "^head" :
        return s[:-5]
    return s

def parse_token(line) :
    idx, token, line = line[0],line[1],line[2:]
    idx = int(idx.split("^")[0]) # in case head is on idx
    tok = Token(token, idx-1, line[:-1])
    return tok

def read_tbk_tree_rec(lines, beg, end, headersize) :
    if len(lines[beg]) == 1 :
        assert(is_xml_beg(lines[beg][0]))
        assert(is_xml_end(lines[end-1][0]))
        label = get_nt_from_xml(lines[beg][0])
        assert(label == get_nt_from_xml(lines[end-1][0]))
        i = beg + 1
        c_beg = []
        counter = 0
        while i < end :
            if counter == 0 :
                c_beg.append(i)
            if is_xml_beg(lines[i][0]) :
                counter += 1
            elif is_xml_end(lines[i][0]) :
                counter -= 1
            i += 1
        children = [ read_tbk_tree_rec(lines, i, j, headersize) for i,j in zip(c_beg[:-1], c_beg[1:]) ]
        #is_head = "^head" in lines[beg][0]
        subtree = Tree(label, children)
        #node = CtbkTree(label, children)
        #node.head = is_head
        #node.idx = min([c.idx for c in node.children])
        #node.children = sorted(node.children, key = lambda x : x.idx)
        return subtree
    else :
        assert(len(lines[beg]) == headersize + 1)
        assert(end == beg + 1)
        return parse_token(lines[beg])

def read_tbk_tree(string, headersize) :
    lines = [ line.strip().split("\t") for line in string.split("\n") if line.strip()]
    return read_tbk_tree_rec(lines, 0, len(lines), headersize)

def read_ctbk_corpus(filename) :
    instream = open(filename, "r")
    header = instream.readline().strip().split()
    assert(header[-1] == "gdeprel")
    Token.header = header[2:-1]
    sentences = instream.read().split("\n\n")

    return [read_tbk_tree(s, len(header)) for s in sentences if s.strip()]

def get_conll(tree):

    tokens = get_yield(tree)

    conll_tokens = []
    for tok in tokens :
        newtok = ["_" for i in range(10)]
        newtok[ID]   = str(tok.i)
        newtok[FORM] = tok.token
        newtok[CPOS] = newtok[FPOS] = tok.features[0]
        newtok[MORPH] = "|".join(sorted(["{}={}".format(a,v) for a,v in zip( Token.header, tok.features[1:] ) if v != "UNDEF"]))
        conll_tokens.append(newtok)
    return conll_tokens

def write_conll(ctree, out):
    for tok in ctree :
        out.write("{}\n".format("\t".join(tok)))


def nltk_tree_to_Tree(nltk_tree):
    # Leaf
    if len(nltk_tree) == 1 and type(nltk_tree[0]) == str:
        idx, token = nltk_tree[0].split("=", 1)
        idx = int(idx)
        return Token(token, idx, [nltk_tree.label()])
    else:
        children = [nltk_tree_to_Tree(child) for child in nltk_tree]
        return Tree(nltk_tree.label(), children)

def read_discbracket_corpus(filename):
    from nltk import Tree as nTree
    with open(filename) as f:
        ctrees = []
        raw = []
        for line in f:
            try:
                ctrees.append(nTree.fromstring(line.strip()))
                raw.append(line)
            except:
                print("?")

    result = [nltk_tree_to_Tree(t) for t in ctrees]
    return result, raw


def transform_discont(discont):
    dict = {}
    span_start = []
    span_end = []
    for dis in discont:
        for span in dis[0]:
            if (span[0], span[1]) not in dict:
                dict[(span[0], span[1])] = len(dict)
                span_start.append(span[0])
                span_end.append(span[1])
    arcs = []
    for dis in discont:
        dis_span = dis[0]
        for i in range(len(dis_span)-1):
            arcs.append( (dict[(dis_span[i][0], dis_span[i][1])], dict[(dis_span[i+1][0], dis_span[i+1][1])], 'NNS'))
        arcs.append((dict[(dis_span[-1][0], dis_span[-1][1])], dict[(dis_span[0][0], dis_span[0][1])], f'THW_{dis[1]}'))

    arcs = list(set(arcs))

    return span_start, span_end, arcs

if __name__ == "__main__":
    input_discbracket_file = None
    pickle_save_dir = None

    def create_dataset(file_name ):
        treebank_1,raw_1 = read_discbracket_corpus(
            file_name)
        word_array = []
        gold_trees = []
        disco = []
        co = []
        raw_trees = []
        count_non = 0
        count_ok = 0
        for tree,raw in zip(treebank_1, raw_1):

            # if not tree.recognizable():
            #     count_non += 1
            #     print(tree.dis())
            #     pass
            try:
                word, pos = tree.get_words()
                cont =  tree.cont()
                discont = tree.dis()
                gold = []
                word_array.append(word)
                gold.append(cont)
                gold.append(discont)
                if len(discont) > 1:
                    disco.append(discont)
                co += cont
                gold_trees.append(gold)
                raw_trees.append(raw)
                count_ok += 1
            except:
                print("????? ")
                continue

        print(count_ok, count_non)



        a = {'word': word_array,
             'gold_tree': gold_trees,
             'raw_tree': raw_trees}

        print(len(word_array))
        with open(pickle_save_dir, "wb") as f:
            pickle.dump(a, f)

    create_dataset(input_discbracket_file)




