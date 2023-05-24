import pdb


class Token:
    # Leaf of a tree
    header = None
    def __init__(self, token, i, features=None):
        self.token = token
        self.features = features # Only used for POS tags for now which should be self.features[0]
        self.i = i
        self.parent = None
        self.span_sorted = [i]

    def get_tag(self):
        if len(self.features) > 0:
            return self.features[0]
        return None

    def set_tag(self, tag):
        self.features[0] = tag

    def is_leaf(self):
        return True

    def is_continuous(self):
        return True

    def get_span(self):
        return {self.i}

    @property
    def span_repr(self):
        return [self.i, self.i+1]

    def return_arc(self):
        return []

    def convert(self):
        pass

    def __str__(self):
        return "({} {}={})".format(self.features[0], self.i, self.token)

    def return_continuous(self):
        return [[self.i, self.i+1]]

    def return_child_span(self):
        return []

    def return_arc(self):
        return []

    def return_arcs(self):
        return []

    def convert(self):
        return []

    def dis(self):
        return []

    def cont(self):
        return []

    def recognizable(self):
        return True

class Tree:
    def __init__(self, label, children):
        self.label = label
        self.children = sorted(children, key = lambda x: min(x.get_span()))
        self.index = -1
        self.span = {i for c in self.children for i in c.get_span()}
        span = list(self.span)
        span.sort()
        self.span_sorted = span
        self.parent = None
        for c in self.children:
            c.parent = self

    # def find_continuous_child(self):

    def recognizable(self):
        num_dis = 0
        for children in self.children:
            if not children.recognizable():
                return False
            if not children.is_continuous():
                num_dis += 1
                if num_dis > 1:
                    return False
        # fan-out > 2,
        try:
            if not self.is_continuous() and len(self.span_repr[0]) > 2:
                return False
        except:
            return False
        return True

    def dis(self):
        dis = []
        #???
        if len(self.span_sorted) == 0:
            return
        if not self.is_continuous():
            dis.append(self.span_repr)
        for child in self.children:
            dis.extend(child.dis())
        return dis

    def cont(self):
        cont = []
        if self.is_continuous():
            cont.append(self.span_repr)
        for child in self.children:
            cont.extend(child.cont())

        return cont

    def return_child_span(self):
        children = []
        for child in self.children:
            children.extend(child.return_continuous())
        return children

    def convert(self):
        s = self.return_arcs()
        for child in self.children:
            s.extend(child.convert())
        return s

    def return_arc(self):
        # arc = []
        # for child in self.children:
        #     if child.is_contiguous():
        #         continue
        #
        #     arc.append()
        if self.is_continuous():
            return []

        else:
            arc = []
            for i in range(len(self.children) - 1):
                arc.append([self.children[i].span_repr, self.children[i+1].span_repr])

            arc.append([self.children[-1].span_repr, self.children[0].span_repr])
            return arc
            # for child in self.children:
            #     arc.append()

    def return_arcs(self):
        arcs = []
        for child in self.children:
            arcs.extend(child.return_arc())
        return arcs


    @property
    def span_repr(self):
        if self.is_continuous():
            return [self.span_sorted[0], self.span_sorted[-1] + 1, self.label]

        else:
            repr = [[self.span_sorted[0], self.span_sorted[0] + 1]]
            for i in range(1, len(self.span_sorted)):
                if self.span_sorted[i] == repr[-1][-1]:
                    repr[-1][-1] = repr[-1][-1]+1
                else:
                    repr.append([self.span_sorted[i], self.span_sorted[i]+1])
            return  [ repr, self.label]

    def is_contiguous(self, array):
        prev = array[0]
        for i in range(1, len(array)):
            if array[i] != prev + 1:
                return False
            prev = array[i]
        return True

    def __len__(self):
        return len(self.span)

    def __le__(self, other):
        return len(self) < len(other)

    def is_continuous(self):
        span = self.span_sorted
        is_continuous = (span[0] + len(span) == span[-1]+1)
        if is_continuous:
            return True
        else:
            return False

    def is_leaf(self):
        assert(self.children != [])
        return False

    def get_span(self):
        return self.span
   
    def get_yield(self, tokens):
        # Updates list of tokens
        for c in self.children:
            if c.is_leaf():
                tokens.append(c)
            else:
                c.get_yield(tokens)

    def merge_unaries(self):
        # Collapse unary nodes
        for c in self.children:
            if not c.is_leaf():
                c.merge_unaries()

        if len(self.children) == 1 and not self.children[0].is_leaf():
            c = self.children[0]
            self.label = "{}@{}".format(self.label, c.label)
            self.children = c.children
            for c in self.children:
                c.parent = self

    def expand_unaries(self):
        # Cancel unary node collapse
        for c in self.children:
            if not c.is_leaf():
                c.expand_unaries()

        if "@" in self.label:
            split_labels = self.label.split("@")
            t = Tree(split_labels[-1], self.children)
            for l in reversed(split_labels[1:-1]):
                t = Tree(l, [t])
            self.label = split_labels[0]
            self.children = [t]
            t.parent = self

    def get_constituents(self, constituents):
        # Update set of constituents
        constituents.add((self.label, tuple(sorted(self.span))))
        for c in self.children:
            if not c.is_leaf():
                c.get_constituents(constituents)

    def __str__(self):
        return "({} {})".format(self.label, " ".join([str(c) for c in self.children]))

    def get_words(self):
        leaves = []
        self.get_yield(leaves)
        leaves = sorted(leaves, key=lambda x:x.i)
        pos = []
        word = []
        for leaf in leaves:
            pos.append(leaf.features)
            word.append(leaf.token)
        return word, pos


def get_yield(tree):
    # Returns list of tokens in the tree (in surface order)
    tokens = []
    tree.get_yield(tokens)
    return sorted(tokens, key = lambda x: min(x.get_span()))

def get_constituents(tree, filter_root=False):
    # Returns a set of constituents in the tree
    # Ignores root labels (from PTB, Negra, and Tiger corpora) if filter_root
    constituents = set()
    tree.get_constituents(constituents)
    if filter_root:
        constituents = {(c, i) for c, i in constituents if c not in {'ROOT', 'VROOT', 'TOP'}}
    return constituents



