import pdb

from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import stripe, diagonal_copy_, diagonal, checkpoint
import torch
import os
import numpy as np
from torch.utils.cpp_extension import load

#
#
try:
    inside_cuda = load(name="inside_on5_cpd_rankspace",
                   sources=["/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_rankspace/lcfrs_cpd_rankspace.cpp", "/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_rankspace/lcfrs_cpd_rankspace.cu"],
                   verbose=True)

except:
    inside_cuda = load(name="inside_on5_cpd_rankspace",
                       sources=["/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_rankspace/lcfrs_cpd_rankspace.cpp", "/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_rankspace/lcfrs_cpd_rankspace.cu"],
                       verbose=True)

class InsideCPD_rankspace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, head_c1, head_c2, left_c, right_c, left_d, right_d,
                cc, dc, head_cd1, head_cd2, head_dd1, head_dd2, unary, root, lens):
        B, L, p = unary.shape
        m = left_c.shape[1] - p
        r1 = head_c1.shape[-1]
        r2 = head_c2.shape[-1]
        r3 = r1
        r4 = r2
        d = 0
        L+=1
        alpha = unary.new_zeros(B, L, L, m).fill_(0)

        # 看显存大小吧..
        alpha_cd = unary.new_zeros(B, L, L, L, L, r2).fill_(0)
        alpha_dd = unary.new_zeros(B, L, L, L, L, r4).fill_(0)

        alpha_lc = unary.new_zeros(B, L, L, r1).fill_(0)
        alpha_rc = unary.new_zeros(B, L, L, r1).fill_(0)
        alpha_ld = unary.new_zeros(B, L, L, r3).fill_(0)
        alpha_rd = unary.new_zeros(B, L, L, r3).fill_(0)
        alpha_cc = unary.new_zeros(B, L, L, r2).fill_(0)
        alpha_dc = unary.new_zeros(B, L, L, r4, 4).fill_(0)

        inside_cuda.forward(head_c1, head_c2, left_c, right_c, left_d, right_d,
                cc, dc, head_cd1, head_cd2, head_dd1, head_dd2, unary, alpha,   alpha_lc, alpha_rc, alpha_ld, alpha_rd,
                alpha_cc, alpha_cd, alpha_dc, alpha_dd,
                B, L, m, p, d, r1, r2, r3, r4)
        to_return1 = (alpha[torch.arange(B), 0, lens] + root)
        to_return2 = to_return1.logsumexp(-1)
        ctx.save_for_backward(
            head_c1, head_c2, left_c, right_c, left_d, right_d,
            cc, dc, head_cd1, head_cd2, head_dd1, head_dd2, unary, root, lens,
            alpha, alpha_lc, alpha_rc, alpha_ld, alpha_rd,
            alpha_cc, alpha_cd, alpha_dc, alpha_dd,
            to_return1, to_return2
        )
        try:
            assert ~to_return2.isnan().any()
        except:
            pdb.set_trace()

        # ctx.save_for_backward(head_c1, head_c2, head_d1, head_d2, left_c, right_c, left_d, right_d,
        #         cc, cd, dc, dd, unary, lens, alpha, alpha_lc, alpha_rc, alpha_ld, alpha_rd,
        #           alpha_cc, alpha_cd, alpha_dc, alpha_dd, to_return1, to_return2)
        #
        return to_return2

    @staticmethod
    def backward(ctx, gwk):
        head_c1, head_c2, left_c, right_c, left_d, right_d,\
        cc, dc, head_cd1, head_cd2, head_dd1, head_dd2, unary, root, lens, \
        alpha, alpha_lc, alpha_rc, alpha_ld, alpha_rd,\
        alpha_cc, alpha_cd, alpha_dc, alpha_dd,   t1, t2 = ctx.saved_tensors

        head_c1_grd = torch.zeros_like(head_c1)
        head_c2_grd = torch.zeros_like(head_c2)
        left_c_grd = torch.zeros_like(left_c)
        left_d_grd = torch.zeros_like(left_d)
        right_c_grd = torch.zeros_like(right_c)
        right_d_grd = torch.zeros_like(right_d)
        cc_grd = torch.zeros_like(cc)
        dc_grd = torch.zeros_like(dc)
        head_cd1_grd = torch.zeros_like(head_cd1)
        head_cd2_grd = torch.zeros_like(head_cd2)
        head_dd1_grd = torch.zeros_like(head_dd1)
        head_dd2_grd = torch.zeros_like(head_dd2)

        unary_grd = torch.zeros_like(unary)
        gradient_root = (t1-t2.unsqueeze(-1)).exp() * gwk.unsqueeze(-1)
        batch_size = alpha.shape[0]
        alpha[torch.arange(batch_size), lens, 0] = gradient_root
        B, L, p = unary.shape
        m = left_c.shape[1] - p
        r1 = head_c1.shape[-1]
        r2 = head_c2.shape[-1]
        r3 = r1
        r4 = r2
        d =  0
        L+=1
        inside_cuda.backward(head_c1, head_c2, left_c, right_c, left_d, right_d,
                cc,  dc, head_cd1, head_cd2, head_dd1, head_dd2, unary,
         head_c1_grd, head_c2_grd, left_c_grd, right_c_grd, left_d_grd, right_d_grd,
         cc_grd, dc_grd, head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd, unary_grd,
        alpha, alpha_lc, alpha_rc, alpha_ld, alpha_rd,
        alpha_cc, alpha_cd, alpha_dc, alpha_dd,
        B, L, m, p, d, r1, r2, r3, r4)

        # print(alpha.transpose(1,2)[torch.ones(L, L).triu()[None, :, :].expand(batch, L, L).bool().cuda()].sum())
        # print(alpha_d.transpose(1,2)[torch.ones(L, L).triu()[None, :, :, None, None].expand(batch, L, L, L, L).bool().cuda()].sum())
        # alpha = alpha.transpose(1, 2)
        # alpha_d = alpha_d.transpose(1, 2)
        try:
            assert head_c1_grd.max() <= 0
            assert head_c2_grd.max() <= 0
            assert left_c_grd.max() <= 0
            assert right_c_grd.max() <= 0
            assert left_d_grd.max() <= 0
            assert right_d_grd.max() <= 0
            assert cc_grd.max() <= 0
            assert dc_grd.max() <= 0
            assert head_cd1_grd.max() <= 0
            assert  head_cd2_grd.max() <= 0

            assert ~head_c1_grd.isinf().any()
            assert ~head_c2_grd.isinf().any()
            assert ~head_dd1_grd.isinf().any()
            assert ~head_dd2_grd.isinf().any()

        except:
            pass

        return  head_c1_grd, head_c2_grd, left_c_grd, right_c_grd, left_d_grd, right_d_grd, \
         cc_grd, dc_grd, head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd, unary_grd,\
            gradient_root, None


class PCFG(PCFG_base):
    def __init__(self):
        super(PCFG, self).__init__()

    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        unary = rules['unary'].contiguous()
        head_c1 = rules['head_c1'].contiguous()
        head_c2 = rules['head_c2'].contiguous()
        left_c = rules['left_c'].contiguous()
        left_d = rules['left_d'].contiguous()
        right_d = rules['right_d'].contiguous()
        right_c = rules['right_c'].contiguous()
        cc = rules['cc'].contiguous()
        dc = rules['dc'].contiguous()
        head_dd1 = rules['head_dd1'].contiguous()
        head_cd1 = rules['head_cd1'].contiguous()
        head_dd2 = rules['head_dd2'].contiguous()
        head_cd2 = rules['head_cd2'].contiguous()

        root = rules['root']
        return {'partition': InsideCPD_rankspace.apply(head_c1, head_c2,  left_c, right_c, left_d, right_d,
                cc, dc,  head_cd1, head_cd2, head_dd1, head_dd2, unary, root, lens)}

    def compute_marginals(self, rules, lens, mbr=True):
        unary = rules['unary']
        head_c1 = rules['head_c1']
        head_c2 = rules['head_c2']
        left_c = rules['left_c']
        left_d = rules['left_d']
        right_d = rules['right_d']
        right_c = rules['right_c']
        cc = rules['cc']
        dc = rules['dc']
        head_dd1 = rules['head_dd1']
        head_cd1 = rules['head_cd1']
        head_dd2 = rules['head_dd2']
        head_cd2 = rules['head_cd2']
        root = rules['root']
        B, L, p = unary.shape
        m = left_c.shape[1] - p
        r1 = head_c1.shape[-1]
        r2 = head_c2.shape[-1]
        r3 = r1
        r4 = r2
        d = 0
        L += 1
        alpha = unary.new_zeros(B, L, L, m).fill_(0)
        # 看显存大小吧..
        alpha_cd = unary.new_zeros(B, L, L, L, L, r2).fill_(0)
        alpha_dd = unary.new_zeros(B, L, L, L, L, r4).fill_(0)
        alpha_lc = unary.new_zeros(B, L, L, r1).fill_(0)
        alpha_rc = unary.new_zeros(B, L, L, r1).fill_(0)
        alpha_ld = unary.new_zeros(B, L, L, r3).fill_(0)
        alpha_rd = unary.new_zeros(B, L, L, r3).fill_(0)
        alpha_cc = unary.new_zeros(B, L, L, r2).fill_(0)
        alpha_dc = unary.new_zeros(B, L, L, r4, 4).fill_(0)
        inside_cuda.forward(head_c1, head_c2, left_c, right_c, left_d, right_d,
                        cc, dc, head_cd1, head_cd2, head_dd1, head_dd2, unary, alpha, alpha_lc, alpha_rc, alpha_ld,
                        alpha_rd,
                        alpha_cc, alpha_cd, alpha_dc, alpha_dd,
                        B, L, m, p, d, r1, r2, r3, r4)
        t1 = (alpha[torch.arange(B), 0, lens] + root)
        t2 = t1.logsumexp(-1)

        partition = t2
        head_c1_grd = torch.zeros_like(head_c1)
        head_c2_grd = torch.zeros_like(head_c2)
        left_c_grd = torch.zeros_like(left_c)
        left_d_grd = torch.zeros_like(left_d)
        right_c_grd = torch.zeros_like(right_c)
        right_d_grd = torch.zeros_like(right_d)
        cc_grd = torch.zeros_like(cc)
        dc_grd = torch.zeros_like(dc)
        head_cd1_grd = torch.zeros_like(head_cd1)
        head_cd2_grd = torch.zeros_like(head_cd2)
        head_dd1_grd = torch.zeros_like(head_dd1)
        head_dd2_grd = torch.zeros_like(head_dd2)

        unary_grd = torch.zeros_like(unary)
        gradient_root = (t1 - t2.unsqueeze(-1)).exp()
        batch_size = alpha.shape[0]
        alpha[torch.arange(batch_size), lens, 0] = gradient_root

        inside_cuda.backward(head_c1, head_c2, left_c, right_c, left_d, right_d,
                         cc, dc, head_cd1, head_cd2, head_dd1, head_dd2, unary,
                         head_c1_grd, head_c2_grd, left_c_grd, right_c_grd, left_d_grd, right_d_grd,
                         cc_grd, dc_grd, head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd, unary_grd,
                         alpha, alpha_lc, alpha_rc, alpha_ld, alpha_rd,
                         alpha_cc, alpha_cd, alpha_dc, alpha_dd,
                         B, L, m, p, d, r1, r2, r3, r4)

        if mbr:
            return alpha, alpha_cd, alpha_dd, partition

        else:
            return head_c1_grd, head_c2_grd, left_c_grd, right_c_grd, left_d_grd, right_d_grd, \
            cc_grd, dc_grd, head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd, unary_grd, gradient_root

    def decode(self, rules, lens, viterbi=False, mbr=False):
        alpha, alpha_cd, alpha_dd, partition = self.compute_marginals(rules, lens, mbr=True)
        B, L = alpha.shape[:2]
        marginal_c = alpha.transpose(1,2).sum(-1)
        marginal_d = alpha_cd.transpose(1,2).sum(-1) + alpha_dd.transpose(1, 2).sum(-1)
        del alpha, alpha_cd,alpha_dd
        alpha_c_mbr = torch.zeros_like(marginal_c)
        alpha_d_mbr = torch.zeros_like(marginal_d)
        # initialize the trivial single_span case.
        alpha_c_mbr[:, torch.arange(L-1), 1+torch.arange(L-1)] = 1
        inside_cuda.argmax(marginal_c, marginal_d, alpha_c_mbr, alpha_d_mbr, B, L)
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
                # 说明这个continuous span是由两个小的continuous span组成而来的.
                else:
                    # flag of ill-nested
                        split = idx
                        assert start<split<end
                        prediction[b_idx][0].append((start, split))
                        prediction[b_idx][0].append((split, end))
                        backtrack(b_idx, start, -1, -1, split)
                        backtrack(b_idx, split, -1, -1, end)

            # discontinuous的span
            else:
                idx = int(alpha_d_mbr[b_idx, gap_start, start, gap_end, end])
                if idx < 0:
                    if idx == -1:
                        prediction[b_idx][0].append((start, gap_start))
                        prediction[b_idx][0].append((gap_end, end))
                        backtrack(b_idx, start, -1, -1, gap_start)
                        backtrack(b_idx, gap_end, -1, -1, end)
                    else:
                        idx = -idx
                        split = int(idx / L)
                        split2 = idx % L
                        assert start < split < gap_start < gap_end < split2 < end
                        prediction[b_idx][1].append((start, split, split2, end))
                        prediction[b_idx][1].append((split, gap_start, gap_end, split2))
                        backtrack(b_idx, start, split, split2, end)
                        backtrack(b_idx, split, gap_start, gap_end, split2)

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