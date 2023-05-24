from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import stripe, diagonal_copy_, diagonal, checkpoint
import torch
import os
import numpy as np
from torch.utils.cpp_extension import load
#
# #
# try:
#     inside_cuda = load(name="lcfrs_on3",
#                    sources=["/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on3_cuda/lcfrs_on3.cpp", "/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on3_cuda/lcfrs_on3.cu"],
#                    verbose=True)
# except:
#     inside_cuda = load(name="lcfrs_on3",
#                    sources=["/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on3_cuda/lcfrs_on3.cpp", "/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on3_cuda/lcfrs_on3.cu"],
#                    verbose=True)





class InsideDisco(torch.autograd.Function):
    @staticmethod
    def forward(ctx, binary,  binary_dc, binary_closed, unary,root, lens):
        B, L, p = unary.shape
        m = binary.shape[-1] - p
        d = binary_dc.shape[-3]
        L+=1
        alpha = unary.new_zeros(B, L, L, m).contiguous()
        alpha_d = unary.new_zeros(B, L, L, L, L, d).contiguous()
        inside_cuda.forward(binary, binary_dc, binary_closed, unary, alpha, alpha_d, B, L, m, p, d)
        to_return1 = alpha[torch.arange(B), 0, lens] + root
        to_return2 = to_return1.logsumexp(-1)
        ctx.save_for_backward(unary, binary, binary_closed, binary_dc, root, lens, alpha, alpha_d, to_return1, to_return2)
        return to_return2

    def backward(ctx, gwk):
        unary, binary, binary_closed, binary_dc, root, lens, alpha, alpha_d, t1, t2 = ctx.saved_tensors
        g_unary, g_binary = unary.new_zeros(*unary.shape), binary.new_zeros(*binary.shape)
        g_binary_closed = binary_closed.new_zeros(*binary_closed.shape)
        g_binary_dc = binary_dc.new_zeros(*binary_dc.shape)
        gradient_root = (t1-t2.unsqueeze(-1)).exp() * gwk.unsqueeze(-1)
        B, L, p = unary.shape
        alpha[torch.arange(B), lens, 0] = gradient_root
        m = binary.shape[-1] - p
        d = binary_dc.shape[-3]
        L+=1
        inside_cuda.backward(binary, binary_dc, binary_closed, unary,
                             g_binary,  g_binary_dc, g_binary_closed, g_unary,
                             alpha, alpha_d, B, L, m, p, d)
        return  g_binary,  g_binary_dc, g_binary_closed, g_unary, gradient_root, None







class LCFRS_ON3(PCFG_base):
    def __init__(self):
        super(LCFRS_ON3, self).__init__()

    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        unary = rules['unary']
        binary = rules['binary']
        binary_closed = rules['binary_closed']
        binary_dc = rules['binary_dc']
        root = rules['root']
        return {'partition': InsideDisco.apply(binary,  binary_dc, binary_closed, unary,root, lens)}

    @torch.enable_grad()
    def decode(self, rules, lens, viterbi=False, mbr=False):
        unary = rules['unary']
        binary = rules['binary']
        binary_closed = rules['binary_closed']
        binary_dc = rules['binary_dc']
        root = rules['root']
        B, L, p = unary.shape
        m = binary.shape[-1] - p
        d = binary_dc.shape[-3]
        L+=1
        alpha = unary.new_zeros(B, L, L, m).fill_(0)
        alpha_d = unary.new_zeros(B, L, L, L, L, d).fill_(0)

        if viterbi:
            partition = InsideDisco.apply(binary,  binary_dc, binary_closed, unary, root, lens)
            inside_cuda.argmax(binary, binary_dc, binary_closed, unary, alpha, alpha_d, B, L, m, p, d)
            logZ, symbol = (alpha[:, 0, -1] + root).max(-1)

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
                        split1 = int(idx / (L * (m + p) * d))
                        split2 = (int(idx / ((m + p) * d))) % (L)
                        assert start < split1 < split2 < end
                        left_symbol = int(idx / (m + p)) % (d)
                        right_symbol = idx % (m + p)
                        prediction[b_idx][1].append((start, split1, split2, end))
                        prediction[b_idx][0].append((split1, split2))
                        backtrack(b_idx, split1, -1, -1, split2, right_symbol)
                        backtrack(b_idx, start, split1, split2, end, left_symbol)
                    # 说明这个continuous span是由两个小的continuous span组成而来的.
                    else:
                        split = int(idx / ((m + p) * (m + p)))
                        assert start<split<end
                        left_symbol = (int(idx / (m + p))) % (m + p)
                        right_symbol = idx % (m + p)
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
                        left_symbol = int(idx / (m + p))
                        right_symbol = idx % (m + p)
                        prediction[b_idx][0].append((start, gap_start))
                        prediction[b_idx][0].append((gap_end, end))
                        backtrack(b_idx, start, -1, -1, gap_start, left_symbol)
                        backtrack(b_idx, gap_end, -1, -1, end, right_symbol)
                    elif idx > 0:
                        # 在这里不能有这种情况 ovo
                        assert NameError
                        split = int(idx / ((m + p) * d * 4))
                        type = idx % 4
                        d_symbol = (int(idx / ((m + p) * 4))) % (d)
                        c_symbol = (int(idx / 4)) % (m + p)
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

        else:
            raise NotImplementedError

        return {'prediction': prediction,
                'partition': partition}


#
#