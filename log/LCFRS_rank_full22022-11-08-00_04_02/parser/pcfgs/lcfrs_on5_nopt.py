from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import stripe, diagonal_copy_, diagonal, checkpoint
import torch
import pdb
import os
import numpy as np
from torch.utils.cpp_extension import load
#
# #
# try:
#     inside_cuda = load(name="inside_on5_nopt",
#                    sources=["/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/cuda_nopt/lcfrs_inside.cpp", "/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/cuda_nopt/lcfrs.cu"],
#                    verbose=True)
#
# except:
#     raise NotImplementedErr



class InsideDisco(torch.autograd.Function):
    @staticmethod
    def forward(ctx, binary, binary_closed, binary_dc, binary_d, unary,root, lens):
        B, L, m = unary.shape
        L+=1
        alpha = unary.new_zeros(B, L, L, m).fill_(0)
        alpha_d = unary.new_zeros(B, L, L, L, L, m).fill_(0)
        alpha[:, torch.arange(L-1), torch.arange(L-1)+1] = unary
        inside_cuda.forward(binary, binary_closed, binary_dc, binary_d, alpha, alpha_d, B, L, m)
        to_return1 = (alpha[torch.arange(B), 0, lens] + root)
        # to_return1 = alpha[:, 0, -1] + root
        to_return2 = to_return1.logsumexp(-1)
        ctx.save_for_backward(unary, binary, binary_closed, binary_dc, binary_d, root, lens, alpha, alpha_d, to_return1, to_return2)
        return to_return2

    @staticmethod
    def backward(ctx, gwk):
        unary, binary, binary_closed, binary_dc, binary_d, root, lens, alpha, alpha_d, t1, t2 = ctx.saved_tensors
        g_unary, g_binary = unary.new_zeros(*unary.shape), binary.new_zeros(*binary.shape)
        g_binary_closed = binary_closed.new_zeros(*binary_closed.shape)
        g_binary_dc = binary_dc.new_zeros(*binary_dc.shape)
        g_binary_d = binary_d.new_zeros(*binary_d.shape)

        gradient_root = (t1-t2.unsqueeze(-1)).exp()
        batch_size = alpha.shape[0]
        alpha[torch.arange(batch_size), lens, 0] = gwk.unsqueeze(-1) * gradient_root
        B, L, m = unary.shape
        L+=1
        inside_cuda.backward(binary, binary_closed, binary_dc, binary_d,
                             g_binary, g_binary_closed, g_binary_dc, g_binary_d,
                             alpha, alpha_d, B, L, m)

        # assert torch.isclose(alpha[:, torch.arange(L-1)+1, torch.arange(L-1)].sum(), lens.float().sum()), f"{alpha[:, torch.arange(L-1)+1, torch.arange(L-1)].sum()}, {lens.float().sum()}"
        return g_binary, g_binary_closed, g_binary_dc, g_binary_d, alpha[:, torch.arange(L-1)+1, torch.arange(L-1)], gwk.unsqueeze(-1) * gradient_root, None



class PCFG(PCFG_base):
    def __init__(self):
        super(PCFG, self).__init__()

    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        unary = rules['unary']
        binary = rules['binary']
        binary_closed = rules['binary_closed']
        binary_dc = rules['binary_dc']
        binary_d = rules['binary_d']
        root = rules['root']
        return {'partition': InsideDisco.apply(binary, binary_closed, binary_dc, binary_d, unary,root, lens)}

    def decode(self, rules, lens, viterbi=False, mbr=False):
        unary = rules['unary']
        binary = rules['binary']
        binary_closed = rules['binary_closed']
        binary_dc = rules['binary_dc']
        binary_d = rules['binary_d']
        root = rules['root']
        B, L, m = unary.shape
        L+=1
        with torch.no_grad():
            partition = InsideDisco.apply(binary, binary_closed, binary_dc, binary_d, unary, root, lens)

        alpha = unary.new_zeros(B, L, L, m).fill_(0)
        alpha_d = unary.new_zeros(B, L, L, L, L, m).fill_(0)
        alpha[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary
        inside_cuda.argmax(binary, binary_closed, binary_dc, binary_d, alpha, alpha_d, B, L, m)
        logZ, symbol = (alpha[torch.arange(B), 0, lens] + root).max(-1)

        prediction = [[[], []] for _ in range(B)]

        def backtrack(b_idx, start, gap_start, gap_end, end, symbol):
            if start + 1  == end:
                return

            nonlocal prediction
            # continuous
            if gap_start == -1:
                idx = int(alpha[b_idx, end, start, symbol])
                # 说明是由一个discontinuous span和一个continuous span合并出来的
                if idx < 0:
                    idx = -idx
                    split1 = int(idx / (L * m * m))
                    split2 = (int(idx / (m * m))) % (L)
                    assert start < split1 < split2 < end
                    left_symbol = int(idx / (m)) % (m)
                    right_symbol = idx % (m )
                    prediction[b_idx][1].append((start, split1, split2, end))
                    prediction[b_idx][0].append((split1, split2))
                    backtrack(b_idx, split1, -1, -1, split2, right_symbol)
                    backtrack(b_idx, start, split1, split2, end, left_symbol)
                # 说明这个continuous span是由两个小的continuous span组成而来的.
                else:
                    split = int(idx / ((m) * (m)))
                    assert start<split<end
                    left_symbol = (int(idx / (m ))) % (m)
                    right_symbol = idx % (m )
                    prediction[b_idx][0].append((start, split))
                    prediction[b_idx][0].append((split, end))
                    backtrack(b_idx, start, -1, -1, split, left_symbol)
                    backtrack(b_idx, split, -1, -1, end, right_symbol)

            # discontinuous的span
            else:
                idx = int(alpha_d[b_idx, gap_start, start, gap_end, end, symbol])
                # 说明这个discontinuous span是由
                if idx < 0:
                    idx = -(idx)
                    left_symbol = int(idx / (m ))
                    right_symbol = idx % (m )
                    prediction[b_idx][0].append((start, gap_start))
                    prediction[b_idx][0].append((gap_end, end))
                    backtrack(b_idx, start, -1, -1, gap_start, left_symbol)
                    backtrack(b_idx, gap_end, -1, -1, end, right_symbol)

                elif idx > 0:
                    split = int(idx / ((m * m) * 4))
                    type = idx % 4
                    d_symbol = (int(idx / ((m ) * 4))) % (m)
                    c_symbol = (int(idx / 4)) % (m)

                    if type == 0:
                        assert start < split
                        assert split < gap_start < gap_end < end
                        prediction[b_idx][0].append((start, split))
                        prediction[b_idx][1].append((split, gap_start, gap_end, end))
                        backtrack(b_idx, start, -1, -1, split, c_symbol)
                        backtrack(b_idx, split, gap_start, gap_end, end, d_symbol)

                    elif type == 1:
                        assert split < gap_start
                        assert start < split < gap_end < end
                        prediction[b_idx][0].append((split, gap_start))
                        prediction[b_idx][1].append((start, split, gap_end, end))
                        backtrack(b_idx, split, -1, -1, gap_start, c_symbol)
                        backtrack(b_idx, start, split, gap_end, end, d_symbol)

                    elif type == 2:
                        assert gap_end < split
                        assert start < gap_start < split < end
                        prediction[b_idx][0].append((gap_end, split))
                        prediction[b_idx][1].append((start, gap_start, split, end))
                        backtrack(b_idx, gap_end, -1, -1, split, c_symbol)
                        backtrack(b_idx, start, gap_start, split, end, d_symbol)

                    else:
                        assert split < end
                        assert start < gap_start < gap_end < split
                        prediction[b_idx][0].append((split, end))
                        prediction[b_idx][1].append((start, gap_start, gap_end, split))
                        backtrack(b_idx, split, -1, -1, end, c_symbol)
                        backtrack(b_idx, start, gap_start, gap_end, split, d_symbol)

                else:
                    assert NameError

        for b_idx in range(B):
            backtrack(b_idx, 0, -1, -1, int(lens[b_idx]), int(symbol[b_idx]))

        return {'prediction': prediction,
                'partition': partition}



