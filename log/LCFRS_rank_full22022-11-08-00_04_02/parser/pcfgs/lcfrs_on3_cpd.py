from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import stripe, diagonal_copy_, diagonal, checkpoint
import torch
import os
import numpy as np
from torch.utils.cpp_extension import load
#
#

try:
    inside_cuda = load(name="inside_on3_cpd",
                   sources=["/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on3_cpd/lcfrs_cpd.cpp", "/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on3_cpd/lcfrs_cpd.cu"],
                   verbose=True)

except:
    inside_cuda = load(name="inside_on3_cpd",
                       sources=["/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on3_cpd/lcfrs_cpd.cpp", "/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on3_cpd/lcfrs_cpd.cu"],
                       verbose=True)

class InsideCPD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, head_c1, head_c2,  head_d1, left_c, right_c, left_d, right_d,
                cc, cd, unary, root, lens):
        B, L, p = unary.shape
        m = left_c.shape[1] - p
        r1 = head_c1.shape[-1]
        r2 = head_c2.shape[-1]
        r3 = head_d1.shape[-1]
        d = cd.shape[1]
        L += 1
        alpha = unary.new_zeros(B, L, L, m).fill_(0)
        alpha_d = unary.new_zeros(B, L, L, L, L, d).fill_(0)

        # 看显存大小吧..
        alpha_cd = unary.new_zeros(B, L, L, L, L, r2).fill_(0)

        alpha_lc = unary.new_zeros(B, L, L, r1).fill_(0)
        alpha_rc = unary.new_zeros(B, L, L, r1).fill_(0)
        alpha_ld = unary.new_zeros(B, L, L, r3).fill_(0)
        alpha_rd = unary.new_zeros(B, L, L, r3).fill_(0)
        alpha_cc = unary.new_zeros(B, L, L, r2).fill_(0)

        inside_cuda.forward(head_c1, head_c2, head_d1, left_c, right_c, left_d, right_d,
                cc, cd,
                unary, alpha,  alpha_d, alpha_lc, alpha_rc, alpha_ld, alpha_rd,
                alpha_cc, alpha_cd,
                B, L, m, p, d, r1, r2, r3)
        to_return1 = (alpha[torch.arange(B), 0, lens] + root)
        to_return2 = to_return1.logsumexp(-1)
        # print(alpha)
        ctx.save_for_backward(head_c1, head_c2, head_d1,  left_c, right_c, left_d, right_d,
                cc, cd,
                unary, alpha,  alpha_d, alpha_lc, alpha_rc, alpha_ld, alpha_rd,
                alpha_cc, alpha_cd, lens, to_return1, to_return2)

        return to_return2

    @staticmethod
    def backward(ctx, gwk):
        head_c1, head_c2, head_d1,  left_c, right_c, left_d, right_d,\
                cc, cd,\
                unary, alpha,  alpha_d, alpha_lc, alpha_rc, alpha_ld, alpha_rd,\
                alpha_cc, alpha_cd, lens, t1, t2 = ctx.saved_tensors

        head_c1_grd = torch.zeros_like(head_c1)
        head_c2_grd = torch.zeros_like(head_c2)

        head_d1_grd = torch.zeros_like(head_d1)
        left_c_grd = torch.zeros_like(left_c)
        left_d_grd = torch.zeros_like(left_d)
        right_c_grd = torch.zeros_like(right_c)
        right_d_grd = torch.zeros_like(right_d)
        cc_grd = torch.zeros_like(cc)
        cd_grd = torch.zeros_like(cd)
        unary_grd = torch.zeros_like(unary)

        gradient_root = (t1-t2.unsqueeze(-1)).exp() * gwk.unsqueeze(-1)
        batch_size = alpha.shape[0]
        alpha[torch.arange(batch_size), lens, 0] = gradient_root
        B, L, p = unary.shape
        m = left_c.shape[1] - p
        r1 = head_c1.shape[-1]
        r2 = head_c2.shape[-1]
        r3 = head_d1.shape[-1]
        d = cd.shape[1]
        L+=1
        inside_cuda.backward(head_c1, head_c2,  head_d1, left_c, right_c, left_d, right_d,
                cc, cd,  unary,
                head_c1_grd, head_c2_grd,  head_d1_grd, left_c_grd, right_c_grd, left_d_grd, right_d_grd,
                cc_grd, cd_grd,  unary_grd,
                alpha,  alpha_d, alpha_lc, alpha_rc, alpha_ld, alpha_rd,
                alpha_cc, alpha_cd,
                B, L, m, p, d, r1, r2, r3)

        # print(alpha.transpose(1,2)[torch.ones(L, L).triu()[None, :, :].expand(batch, L, L).bool().cuda()].sum())
        # print(alpha_d.transpose(1,2)[torch.ones(L, L).triu()[None, :, :, None, None].expand(batch, L, L, L, L).bool().cuda()].sum())
        # alpha = alpha.transpose(1, 2)
        # alpha_d = alpha_d.transpose(1, 2)

        return head_c1_grd, head_c2_grd,  head_d1_grd,  left_c_grd, right_c_grd, left_d_grd, right_d_grd,\
                cc_grd, cd_grd,  unary_grd, gradient_root,  None



class PCFG(PCFG_base):
    def __init__(self):
        super(PCFG, self).__init__()

    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        unary = rules['unary']
        head_c1 = rules['head_c1']
        head_d1 = rules['head_d1']
        head_c2 = rules['head_c2']

        left_c = rules['left_c']
        left_d = rules['left_d']
        right_d = rules['right_d']
        right_c = rules['right_c']
        cc = rules['cc']
        cd = rules['cd']
        root = rules['root']
        return {'partition': InsideCPD.apply(head_c1, head_c2, head_d1,  left_c, right_c, left_d, right_d,
                cc, cd, unary, root, lens)}


    def decode(self, rules, lens, viterbi=False, mbr=False):
        unary = rules['unary']
        head_c1 = rules['head_c1']
        head_d1 = rules['head_d1']
        head_c2 = rules['head_c2']

        left_c = rules['left_c']
        left_d = rules['left_d']
        right_d = rules['right_d']
        right_c = rules['right_c']
        cc = rules['cc']
        cd = rules['cd']
        root = rules['root']
        B, L, p = unary.shape
        B, L, p = unary.shape
        m = left_c.shape[1] - p
        r1 = head_c1.shape[-1]
        r2 = head_c2.shape[-1]
        r3 = head_d1.shape[-1]
        d = cd.shape[1]
        L += 1
        alpha = unary.new_zeros(B, L, L, m).fill_(0)
        alpha_d = unary.new_zeros(B, L, L, L, L, d).fill_(0)

        # 看显存大小吧..
        alpha_cd = unary.new_zeros(B, L, L, L, L, r2).fill_(0)

        alpha_lc = unary.new_zeros(B, L, L, r1).fill_(0)
        alpha_rc = unary.new_zeros(B, L, L, r1).fill_(0)
        alpha_ld = unary.new_zeros(B, L, L, r3).fill_(0)
        alpha_rd = unary.new_zeros(B, L, L, r3).fill_(0)
        alpha_cc = unary.new_zeros(B, L, L, r2).fill_(0)

        inside_cuda.forward(head_c1, head_c2, head_d1, left_c, right_c, left_d, right_d,
                cc, cd,
                unary, alpha,  alpha_d, alpha_lc, alpha_rc, alpha_ld, alpha_rd,
                alpha_cc, alpha_cd,
                B, L, m, p, d, r1, r2, r3)
        t1 = (alpha[torch.arange(B), 0, lens] + root)
        t2 = t1.logsumexp(-1)
        partition = t2

        head_c1_grd = torch.zeros_like(head_c1)
        head_c2_grd = torch.zeros_like(head_c2)

        head_d1_grd = torch.zeros_like(head_d1)
        left_c_grd = torch.zeros_like(left_c)
        left_d_grd = torch.zeros_like(left_d)
        right_c_grd = torch.zeros_like(right_c)
        right_d_grd = torch.zeros_like(right_d)
        cc_grd = torch.zeros_like(cc)
        cd_grd = torch.zeros_like(cd)
        unary_grd = torch.zeros_like(unary)

        gradient_root = (t1-t2.unsqueeze(-1)).exp()
        batch_size = alpha.shape[0]
        alpha[torch.arange(batch_size), lens, 0] = gradient_root
        B, L, p = unary.shape
        m = left_c.shape[1] - p
        r1 = head_c1.shape[-1]
        r2 = head_c2.shape[-1]
        r3 = head_d1.shape[-1]
        d = cd.shape[1]
        L+=1
        inside_cuda.backward(head_c1, head_c2,  head_d1,  left_c, right_c, left_d, right_d,
                cc, cd,  unary,
                head_c1_grd, head_c2_grd,  head_d1_grd,  left_c_grd, right_c_grd, left_d_grd, right_d_grd,
                cc_grd, cd_grd,  unary_grd,
                alpha,  alpha_d, alpha_lc, alpha_rc, alpha_ld, alpha_rd,
                alpha_cc, alpha_cd,
                B, L, m, p, d, r1, r2, r3)
        del alpha_lc, alpha_rc, alpha_ld, alpha_rd, alpha_cc, alpha_cd
        marginal_c = alpha.transpose(1,2).sum(-1)
        marginal_d = alpha_d.transpose(1,2).sum(-1)
        alpha_c_mbr = torch.zeros_like(marginal_c)
        alpha_d_mbr = torch.zeros_like(marginal_d)
        # initialize the trivial single_span case.
        alpha_c_mbr[:, torch.arange(L-1), 1+torch.arange(L-1)] = 1
        inside_cuda.argmax(marginal_c, marginal_d, alpha_c_mbr, alpha_d_mbr, B, L)

        # alpha_mask = alpha.new_zeros(batch_size, L, L)
        # alpha_mask[:, torch.arange(L-1), 1+torch.arange(L-1)] = 1
        # alpha_mask[:, 0, -1] = 1
        # alpha_d_mask = alpha.new_zeros(batch_size, L, L, L, L)
        prediction = [[[], []] for _ in range(B)]

        def backtrack(b_idx, start, gap_start, gap_end, end):
            if start + 1  == end:
                return

            nonlocal prediction
            # continuous
            if gap_start == -1:
                idx = int(alpha_c_mbr[b_idx, end, start])

                if idx < 0:
                    idx = -idx
                    split1 = int(idx / L)
                    split2 = idx % (L)
                    assert start < split1 < split2 < end
                    prediction[b_idx][1].append((start, split1, split2, end))
                    prediction[b_idx][0].append((split1, split2))
                    backtrack(b_idx, split1, -1, -1, split2)
                    backtrack(b_idx, start, split1, split2, end)

                else:
                    # 说明这个continuous span是由两个小的continuous span组成而来的.
                    if idx < L:
                        split = idx
                        assert start<split<end
                        prediction[b_idx][0].append((start, split))
                        prediction[b_idx][0].append((split, end))
                        backtrack(b_idx, start, -1, -1, split)
                        backtrack(b_idx, split, -1, -1, end)


                    # ill-nested [a, b, c, d], [b, c, d, e] -> [a, e]
                    else:
                        split1 = int(idx / (L*L))
                        split2 = int(idx / L) % L
                        split3 = idx % L
                        assert start < split1 < split2 < split3 < end
                        prediction[b_idx][1].append((start, split1, split2, split3))
                        prediction[b_idx][1].append((split1, split2, split3, end))
                        backtrack(b_idx, start, split1, split2, split3)
                        backtrack(b_idx, split1, split2, split3, end)

            # discontinuous的span
            else:
                idx = int(alpha_d_mbr[b_idx, gap_start, start, gap_end, end])
                if idx < 0:
                    prediction[b_idx][0].append((start, gap_start))
                    prediction[b_idx][0].append((gap_end, end))
                    backtrack(b_idx, start, -1, -1, gap_start)
                    backtrack(b_idx, gap_end, -1, -1, end)

                elif idx > 0:
                    type = int(idx / L)
                    split = idx % L
                    if type == 0:
                        assert start < split
                        assert split < gap_start < gap_end < end
                        prediction[b_idx][0].append((start, split))
                        prediction[b_idx][1].append((split, gap_start, gap_end, end))
                        backtrack(b_idx, start, -1, -1, split)
                        backtrack(b_idx, split, gap_start, gap_end, end)

                    elif type == 1:
                        assert split < gap_start
                        assert start < split < gap_end < end
                        prediction[b_idx][0].append((split, gap_start))
                        prediction[b_idx][1].append((start, split, gap_end, end))
                        backtrack(b_idx, split, -1, -1, gap_start)
                        backtrack(b_idx, start, split, gap_end, end)

                    elif type == 2:
                        assert gap_end < split
                        assert start < gap_start < split < end
                        prediction[b_idx][0].append((gap_end, split))
                        prediction[b_idx][1].append((start, gap_start, split, end))
                        backtrack(b_idx, gap_end, -1, -1, split)
                        backtrack(b_idx, start, gap_start, split, end)

                    else:
                        assert split < end
                        assert start < gap_start < gap_end < split
                        prediction[b_idx][0].append((split, end))
                        prediction[b_idx][1].append((start, gap_start, gap_end, split))
                        backtrack(b_idx, split, -1, -1, end)
                        backtrack(b_idx, start, gap_start, gap_end, split)
                else:
                    assert NameError

        for b_idx in range(B):
            backtrack(b_idx, 0, -1, -1, int(lens[b_idx]))

        return {'prediction': prediction,
                    'partition': partition}


#
#