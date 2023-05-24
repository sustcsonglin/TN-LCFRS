import torch
from .trees import Tree, Token
from parser.lcfrs_triton.argmax import argmax_on5_wn




def mbr_decoding(marginal_c, marginal_d, raw_word, lens):    
    B = marginal_c.shape[0] 
    L = marginal_c.shape[1]
    alpha_c_mbr = torch.zeros_like(marginal_c)
    alpha_d_mbr = torch.zeros_like(marginal_d)        
    # initialize the trivial single_span case.
    # alpha_c_mbr[:, torch.arange(L-1), 1+torch.arange(L-1)] = 1

    prediction = [[[], []] for _ in range(B)]

    argmax_on5_wn(alpha_c_mbr, alpha_d_mbr, marginal_c, marginal_d)
    
    def backtrack(b_idx, start, gap_start, gap_end, end, root=False):
        if start + 1  == end:
            token_node = [Token(raw_word[b_idx][start], start, 'PP')]
            if root:
                token_node = Tree("NP", token_node)
            return token_node

        nonlocal prediction
        # continuous parent span
        if gap_start == -1:
            idx = int(alpha_c_mbr[b_idx, end, start])
            # [start, end] = [start, split1, split2, end] + [split1, split2]
            if idx < 0:
                idx = -idx
                split1 = int(idx / L)
                split2 = idx % (L)
                assert start < split1 < split2 < end
                prediction[b_idx][1].append((start, split1, split2, end))
                prediction[b_idx][0].append((split1, split2))
                child1 = backtrack(b_idx, split1, -1, -1, split2)
                child2 = backtrack(b_idx, start, split1, split2, end)
            # [start, end] = [start, split] + [split end]
            else:
                split = idx
                assert start<split<end, f"({start}, {split}, {end})"
                prediction[b_idx][0].append((start, split))
                prediction[b_idx][0].append((split, end))
                child1 = backtrack(b_idx, start, -1, -1, split)
                child2 = backtrack(b_idx, split, -1, -1, end)
        # discontinuous parent span
        else:
            idx = int(alpha_d_mbr[b_idx, gap_start, start, gap_end, end])
            if idx < 0:
                if idx == -1:
                    prediction[b_idx][0].append((start, gap_start))
                    prediction[b_idx][0].append((gap_end, end))
                    child1 = backtrack(b_idx, start, -1, -1, gap_start)
                    child2 = backtrack(b_idx, gap_end, -1, -1, end)
                else:
                    idx = -idx
                    split = int(idx / L)
                    split2 = idx % L
                    assert start < split < gap_start < gap_end < split2 < end
                    prediction[b_idx][1].append((start, split, split2, end))
                    prediction[b_idx][1].append((split, gap_start, gap_end, split2))
                    child1 = backtrack(b_idx, start, split, split2, end)
                    child2 = backtrack(b_idx, split, gap_start, gap_end, split2)
            elif idx > 0:
                type = int(idx / L)
                split = idx % L
                if type == 0:
                    assert start < split
                    assert split < gap_start < gap_end < end
                    prediction[b_idx][0].append((start, split))
                    prediction[b_idx][1].append((split, gap_start, gap_end, end))
                    child1 = backtrack(b_idx, start, -1, -1, split)
                    child2 = backtrack(b_idx, split, gap_start, gap_end, end)
                elif type == 1:
                    assert split < gap_start
                    assert start < split < gap_end < end
                    prediction[b_idx][0].append((split, gap_start))
                    prediction[b_idx][1].append((start, split, gap_end, end))
                    child1 = backtrack(b_idx, split, -1, -1, gap_start)
                    child2 = backtrack(b_idx, start, split, gap_end, end)
                elif type == 2:
                    assert gap_end < split
                    assert start < gap_start < split < end
                    prediction[b_idx][0].append((gap_end, split))
                    prediction[b_idx][1].append((start, gap_start, split, end))
                    child1 =  backtrack(b_idx, gap_end, -1, -1, split)
                    child2 =  backtrack(b_idx, start, gap_start, split, end)
                else:
                    assert split < end
                    assert start < gap_start < gap_end < split
                    prediction[b_idx][0].append((split, end))
                    prediction[b_idx][1].append((start, gap_start, gap_end, split))
                    child1 = backtrack(b_idx, split, -1, -1, end)
                    child2 = backtrack(b_idx, start, gap_start, gap_end, split)
            else:
                assert NameError
        node = Tree("NT", child1 + child2)
        if root:
            return node
        else:
            return [node]

    predicted_trees = []
    for b_idx in range(B):
        tree = backtrack(b_idx, 0, -1, -1, int(lens[b_idx]), root=True)
        predicted_trees.append(str(tree))

    return prediction, predicted_trees
