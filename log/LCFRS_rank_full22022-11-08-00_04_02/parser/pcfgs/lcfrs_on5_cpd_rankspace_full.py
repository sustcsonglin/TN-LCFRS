import pdb

from parser.pcfgs.pcfgs import PCFG_base
from .lcfrs_on5_cpd_rankspace import inside_cuda as inside_cuda2
from parser.pcfgs.fn import stripe, diagonal_copy_, diagonal, checkpoint
import torch
import os
import numpy as np
from torch.utils.cpp_extension import load
#
#
try:
    inside_cuda = load(name="Ysl",
                   sources=["/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_rankspace2/lcfrs_cpd_rankspace.cpp", "/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_rankspace2/lcfrs_cpd_rankspace.cu"],
                   verbose=True)

except:
    inside_cuda = load(name="inside_on5_cpd_rankspace",
                       sources=["/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_rankspace/lcfrs_cpd_rankspace.cpp", "/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_rankspace/lcfrs_cpd_rankspace.cu"],
                       verbose=True)

class InsideCPD_rankspace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, head_cl1, head_cr1, head_cc1, head_dc1,
                     head_cl2, head_cr2, head_cc2, head_dc2,
                     head_cd1, head_cd2, head_dd1, head_dd2,
                     unary_l, unary_r, unary_c, unary_d,
                     root_c, root_d, lens):

        B, L, r1 = unary_l.shape
        r2 = unary_c.shape[-1]
        L+=1
        alpha_l = unary_l.new_zeros(B, L, L, r1).fill_(0).contiguous()
        alpha_r = unary_l.new_zeros(B, L, L, r1).fill_(0).contiguous()
        alpha_cc = unary_l.new_zeros(B, L, L, r2).fill_(0).contiguous()
        alpha_dc = unary_l.new_zeros(B, L, L, 4, r2).fill_(0).contiguous()
        #
        alpha_cd = unary_l.new_zeros(B, L, L, L, L, r2).contiguous()
        # alpha_cd = torch.arange(B*L*L*L*L*r2).reshape(B, L, L, L, L, r2).float().cuda()
        alpha_dd = unary_l.new_zeros(B, L, L, L, L, r2).fill_(0).contiguous()

        alpha_l[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_l
        alpha_r[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_r
        alpha_cc[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_c
        alpha_dc[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_d

        alpha_l = alpha_l.contiguous()
        alpha_r = alpha_r.contiguous()
        alpha_cc = alpha_cc.contiguous()
        alpha_dc = alpha_dc.contiguous()

        inside_cuda.forward(
            head_cl1, head_cr1, head_cc1, head_dc1,
            head_cl2.contiguous(), head_cr2, head_cc2, head_dc2.contiguous(),
            head_cd1.contiguous(), head_cd2.contiguous(), head_dd1, head_dd2,
            root_c.contiguous(), root_d.contiguous(),
            alpha_l.contiguous(), alpha_r.contiguous(), alpha_cc.contiguous(), alpha_cd.contiguous(),
            alpha_dc.contiguous(), alpha_dd.contiguous(), B, L, r1, r2
        )
        ctx.save_for_backward(
            head_cl1, head_cr1, head_cc1, head_dc1,
            head_cl2, head_cr2, head_cc2, head_dc2,
            head_cd1, head_cd2, head_dd1, head_dd2,
            root_c, root_d,
            alpha_l.contiguous(), alpha_r.contiguous(), alpha_cc.contiguous(), alpha_cd.contiguous(),
            alpha_dc.contiguous(), alpha_dd.contiguous(),
        )

        return alpha_l[:, 0, -1, 0].detach()


    @staticmethod
    def backward(ctx, gwk):
        head_cl1, head_cr1, head_cc1, head_dc1, \
        head_cl2, head_cr2, head_cc2, head_dc2,\
        head_cd1, head_cd2, head_dd1, head_dd2,\
        root_c, root_d,\
        alpha_l, alpha_r, alpha_cc, alpha_cd,\
        alpha_dc, alpha_dd = ctx.saved_tensors

        B, L = alpha_l.shape[:2]
        r1 = alpha_l.shape[-1]
        r2 = alpha_cc.shape[-1]


        head_cl1_grd = torch.zeros_like(head_cl1)
        head_cl2_grd = torch.zeros_like(head_cl2)
        head_cr1_grd = torch.zeros_like(head_cr1)
        head_cr2_grd = torch.zeros_like(head_cr2)
        head_cc1_grd = torch.zeros_like(head_cc1)
        head_cc2_grd = torch.zeros_like(head_cc2)
        head_cd1_grd = torch.zeros_like(head_cd1)
        head_cd2_grd = torch.zeros_like(head_cd2)
        head_dd1_grd = torch.zeros_like(head_dd1)
        head_dd2_grd = torch.zeros_like(head_dd2)
        head_dc1_grd = torch.zeros_like(head_dc1)
        head_dc2_grd = torch.zeros_like(head_dc2)

        root_c_grd = torch.zeros_like(root_c)
        root_d_grd = torch.zeros_like(root_d)

        alpha_l[:, -1, 0, 0] = gwk

        inside_cuda.backward(
            head_cl1, head_cr1, head_cc1, head_dc1,
            head_cl2, head_cr2, head_cc2, head_dc2,
            head_cd1, head_cd2, head_dd1, head_dd2,
            root_c, root_d,

            head_cl1_grd, head_cr1_grd, head_cc1_grd, head_dc1_grd,
            head_cl2_grd, head_cr2_grd, head_cc2_grd, head_dc2_grd,
            head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd,
            root_c_grd, root_d_grd,

            alpha_l.contiguous(), alpha_r.contiguous(), alpha_cc.contiguous(), alpha_cd.contiguous(), alpha_dc.contiguous(), alpha_dd.contiguous(),
            B, L, r1, r2
        )

        assert ~torch.isnan(head_cl1_grd).any()
        assert ~torch.isinf(head_cl1_grd).any()
        assert ~torch.isnan(head_cr1_grd).any()
        assert ~torch.isinf(head_cr1_grd).any()
        assert ~torch.isnan(head_cc1_grd).any()
        assert ~torch.isinf(head_cc1_grd).any()
        assert ~torch.isnan(head_dc1_grd).any()
        assert ~torch.isinf(head_dc1_grd).any()
        assert ~torch.isnan(head_cl2_grd).any()
        assert ~torch.isinf(head_cl2_grd).any()
        assert ~torch.isnan(head_cr2_grd).any()
        assert ~torch.isinf(head_cr2_grd).any()
        assert ~torch.isnan(head_cc2_grd).any()
        assert ~torch.isinf(head_cc2_grd).any()
        assert ~torch.isnan(head_dc2_grd).any()
        assert ~torch.isinf(head_dc2_grd).any()
        assert ~torch.isnan(head_cd1_grd).any()
        assert ~torch.isinf(head_cd1_grd).any()
        assert ~torch.isnan(head_cd2_grd).any()
        assert ~torch.isinf(head_cd2_grd).any()
        assert ~torch.isnan(head_dd1_grd).any()
        assert ~torch.isinf(head_dd1_grd).any()
        assert ~torch.isnan(head_dd2_grd).any()
        assert ~torch.isinf(head_dd2_grd).any()
        #
        # assert head_cl1_grd.max() <= 0
        # assert head_cr1_grd.max() <= 0
        # assert head_cc1_grd.max() <= 0
        # assert head_dc1_grd.max() <= 0
        # assert head_cd1_grd.max() <= 0
        # assert head_dd1_grd.max() <= 0
        #


        return  head_cl1_grd, head_cr1_grd, head_cc1_grd, head_dc1_grd, \
                head_cl2_grd, head_cr2_grd, head_cc2_grd, head_dc2_grd, \
                head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd, \
                alpha_l[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_r[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_cc[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_dc[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                root_c_grd, root_d_grd, None


class PCFG(PCFG_base):
    def __init__(self):
        super(PCFG, self).__init__()

    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        head_dd1 = rules['head_dd1'].contiguous()
        head_cd1 = rules['head_cd1'].contiguous()
        head_dd2 = rules['head_dd2'].contiguous()
        head_cd2 = rules['head_cd2'].contiguous()

        unary_l = rules['unary_l'].contiguous()
        unary_r = rules['unary_r'].contiguous()
        unary_c = rules['unary_c'].contiguous()
        unary_d = rules['unary_d'].contiguous()

        head_cl1 = rules['head_cl1'].contiguous()
        head_cl2 = rules['head_cl2'].contiguous()
        head_cr1 = rules['head_cr1'].contiguous()
        head_cr2 = rules['head_cr2'].contiguous()
        head_cc1 = rules['head_cc1'].contiguous()
        head_cc2 = rules['head_cc2'].contiguous()
        head_dc1 = rules['head_dc1'].contiguous()
        head_dc2 = rules['head_dc2'].contiguous()

        root_c = rules['root_c'].contiguous()
        root_d = rules['root_d'].contiguous()

        return {'partition': InsideCPD_rankspace.apply(head_cl1, head_cr1, head_cc1, head_dc1,
                     head_cl2, head_cr2, head_cc2, head_dc2,
                     head_cd1, head_cd2, head_dd1, head_dd2,
                     unary_l, unary_r, unary_c, unary_d,
                     root_c, root_d, lens)}


    def decode(self, rules, lens, viterbi=False, mbr=False):
        head_dd1 = rules['head_dd1'].contiguous()
        head_cd1 = rules['head_cd1'].contiguous()
        head_dd2 = rules['head_dd2'].contiguous()
        head_cd2 = rules['head_cd2'].contiguous()

        unary_l = rules['unary_l'].contiguous()
        unary_r = rules['unary_r'].contiguous()
        unary_c = rules['unary_c'].contiguous()
        unary_d = rules['unary_d'].contiguous()

        head_cl1 = rules['head_cl1'].contiguous()
        head_cl2 = rules['head_cl2'].contiguous()
        head_cr1 = rules['head_cr1'].contiguous()
        head_cr2 = rules['head_cr2'].contiguous()
        head_cc1 = rules['head_cc1'].contiguous()
        head_cc2 = rules['head_cc2'].contiguous()
        head_dc1 = rules['head_dc1'].contiguous()
        head_dc2 = rules['head_dc2'].contiguous()

        root_c = rules['root_c'].contiguous()
        root_d = rules['root_d'].contiguous()

        # forward
        B, L, r1 = unary_l.shape
        r2 = unary_c.shape[-1]
        L+=1
        alpha_l = unary_l.new_zeros(B, L, L, r1).fill_(0).contiguous()
        alpha_r = unary_l.new_zeros(B, L, L, r1).fill_(0).contiguous()
        alpha_cc = unary_l.new_zeros(B, L, L, r2).fill_(0).contiguous()
        alpha_dc = unary_l.new_zeros(B, L, L, 4, r2).fill_(0).contiguous()
        #
        alpha_cd = unary_l.new_zeros(B, L, L, L, L, r2).contiguous()
        # alpha_cd = torch.arange(B*L*L*L*L*r2).reshape(B, L, L, L, L, r2).float().cuda()
        alpha_dd = unary_l.new_zeros(B, L, L, L, L, r2).fill_(0).contiguous()

        alpha_l[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_l
        alpha_r[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_r
        alpha_cc[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_c
        alpha_dc[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_d

        inside_cuda.forward(
            head_cl1, head_cr1, head_cc1, head_dc1,
            head_cl2, head_cr2, head_cc2, head_dc2,
            head_cd1, head_cd2, head_dd1, head_dd2,
            root_c, root_d,
            alpha_l, alpha_r, alpha_cc, alpha_cd,
            alpha_dc, alpha_dd, B, L, r1, r2
        )


        partition = alpha_l[:, 0, -1, 0].detach()

        # backward
        head_cl1_grd = torch.zeros_like(head_cl1)
        head_cl2_grd = torch.zeros_like(head_cl2)
        head_cr1_grd = torch.zeros_like(head_cr1)
        head_cr2_grd = torch.zeros_like(head_cr2)
        head_cc1_grd = torch.zeros_like(head_cc1)
        head_cc2_grd = torch.zeros_like(head_cc2)
        head_cd1_grd = torch.zeros_like(head_cd1)
        head_cd2_grd = torch.zeros_like(head_cd2)
        head_dd1_grd = torch.zeros_like(head_dd1)
        head_dd2_grd = torch.zeros_like(head_dd2)
        head_dc1_grd = torch.zeros_like(head_dc1)
        head_dc2_grd = torch.zeros_like(head_dc2)

        root_c_grd = torch.zeros_like(root_c)
        root_d_grd = torch.zeros_like(root_d)

        alpha_l[:, -1, 0, 0] = 1



        inside_cuda.backward(
            head_cl1, head_cr1, head_cc1, head_dc1,
            head_cl2, head_cr2, head_cc2, head_dc2,
            head_cd1, head_cd2, head_dd1, head_dd2,
            root_c, root_d,

            head_cl1_grd, head_cr1_grd, head_cc1_grd, head_dc1_grd,
            head_cl2_grd, head_cr2_grd, head_cc2_grd, head_dc2_grd,
            head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd,
            root_c_grd, root_d_grd,

            alpha_l, alpha_r, alpha_cc, alpha_cd, alpha_dc, alpha_dd,
            B, L, r1, r2
        )


        # span marginals.
        marginal_c = alpha_l.transpose(1, 2).sum(-1) + alpha_r.transpose(1, 2).sum(-1) \
                    + alpha_cc.transpose(1, 2).sum(-1) + alpha_dc.transpose(1, 2).sum([-1, -2])

        marginal_d = alpha_cd.transpose(1,2).sum(-1) + alpha_dd.transpose(1, 2).sum(-1)
        alpha_c_mbr = torch.zeros_like(marginal_c)
        alpha_d_mbr = torch.zeros_like(marginal_d)
        # initialize the trivial single_span case.



        alpha_c_mbr[:, torch.arange(L-1), 1+torch.arange(L-1)] = 1

        inside_cuda2.argmax(marginal_c, marginal_d, alpha_c_mbr, alpha_d_mbr, B, L)
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