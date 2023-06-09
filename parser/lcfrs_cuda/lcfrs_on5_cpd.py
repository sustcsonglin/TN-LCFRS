import pdb
import torch
import os
import numpy as np
from torch.utils.cpp_extension import load
from parser.mbr_decoding import mbr_decoding

try:
    inside_cuda = load(name="lcfrs_on5_rank",
                   sources=["parser/lcfrs_cuda/cuda_lcfrs_on5_rank/lcfrs_cpd.cpp", "parser/lcfrs_cuda/cuda_lcfrs_on5_rank/lcfrs_cpd.cu"],
                   verbose=True)
except:
    pass 


class InsideCPD_rankspace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, head_cl1, head_cr1,
                     head_dl1, head_dr1,
                     head_cc1, head_dc1,
                     head_cl2, head_cr2,
                     head_dl2, head_dr2,
                     head_cc2, head_dc2,
                     head_cd1, head_cd2, head_dd1, head_dd2,
                     unary_l, unary_r,
                     unary_ld, unary_rd,
                     unary_c, unary_d,
                     root_c, root_d, lens):

        B, L, r1 = unary_l.shape
        r2 = unary_c.shape[-1]
        r3 = unary_ld.shape[-1]
        r4 = unary_d.shape[-1]
        # pdb.set_trace()

        L+=1
        alpha_l = unary_l.new_zeros(B, L, L, r1).fill_(0).contiguous()
        alpha_r = unary_l.new_zeros(B, L, L, r1).fill_(0).contiguous()
        alpha_cc = unary_l.new_zeros(B, L, L, r2).fill_(0).contiguous()
        alpha_dc = unary_l.new_zeros(B, L, L, 4, r4).fill_(0).contiguous()

        alpha_ld = unary_l.new_zeros(B, L, L, r3).fill_(0).contiguous()
        alpha_rd = unary_l.new_zeros(B, L, L, r3).fill_(0).contiguous()

        alpha_cd = unary_l.new_zeros(B, L, L, L, L, r2).fill_(0).contiguous()
        alpha_dd = unary_l.new_zeros(B, L, L, L, L, r4).fill_(0).contiguous()

        alpha_tmp_c1 = unary_l.new_zeros(B, L, L, r1).fill_(0).contiguous()
        alpha_tmp_c2 = unary_l.new_zeros(B, L, L, r2).fill_(0).contiguous()

        alpha_l[:,  torch.arange(L-1), torch.arange(L-1) + 1] = unary_l
        alpha_ld[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_ld
        alpha_r[:,  torch.arange(L-1),  torch.arange(L-1) + 1] = unary_r
        alpha_rd[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_rd
        alpha_cc[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_c
        alpha_dc[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_d

        inside_cuda.forward(
            head_cl1, head_cr1,
            head_dl1, head_dr1,
            head_cc1, head_dc1,
            head_cl2, head_cr2,
            head_dl2, head_dr2,
            head_cc2, head_dc2,
            head_cd1, head_cd2, head_dd1, head_dd2,
            root_c, root_d,
            alpha_l, alpha_r,
            alpha_ld, alpha_rd,
            alpha_cc, alpha_cd,
            alpha_dc, alpha_dd,
            alpha_tmp_c1, alpha_tmp_c2,
            B, L, r1, r2, r3, r4
        )

        # pdb.set_trace()

        ctx.save_for_backward(
            head_cl1, head_cr1,
            head_dl1, head_dr1,
            head_cc1, head_dc1,
            head_cl2, head_cr2,
            head_dl2, head_dr2,
            head_cc2, head_dc2,
            head_cd1, head_cd2, head_dd1, head_dd2,
            root_c, root_d,
            alpha_l, alpha_r,
            alpha_ld, alpha_rd,
            alpha_cc, alpha_cd,
            alpha_dc, alpha_dd,
            alpha_tmp_c1, alpha_tmp_c2,
        )
        return alpha_l[:, 0, -1, 0].detach()


    @staticmethod
    def backward(ctx, gwk):

        head_cl1, head_cr1,\
        head_dl1, head_dr1,\
        head_cc1, head_dc1,\
        head_cl2, head_cr2,\
        head_dl2, head_dr2,\
        head_cc2, head_dc2,\
        head_cd1, head_cd2, head_dd1, head_dd2,\
        root_c, root_d,\
        alpha_l, alpha_r,\
        alpha_ld, alpha_rd,\
        alpha_cc, alpha_cd,\
        alpha_dc, alpha_dd,\
        alpha_tmp_c1, alpha_tmp_c2 = ctx.saved_tensors

        B, L = alpha_l.shape[:2]
        r1 = alpha_l.shape[-1]
        r2 = alpha_cd.shape[-1]
        r3 = alpha_ld.shape[-1]
        r4 = alpha_dd.shape[-1]

        head_cl1_grd = torch.zeros_like(head_cl1)
        head_dl1_grd = torch.zeros_like(head_dl1)
        head_cl2_grd = torch.zeros_like(head_cl2)
        head_dl2_grd = torch.zeros_like(head_dl2)

        head_cr1_grd = torch.zeros_like(head_cr1)
        head_dr1_grd = torch.zeros_like(head_dr1)
        head_cr2_grd = torch.zeros_like(head_cr2)
        head_dr2_grd = torch.zeros_like(head_dr2)

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
            head_cl1, head_cr1,
            head_dl1, head_dr1,
            head_cc1, head_dc1,
            head_cl2, head_cr2,
            head_dl2, head_dr2,
            head_cc2, head_dc2,
            head_cd1, head_cd2,
            head_dd1, head_dd2,
            root_c, root_d,

            head_cl1_grd, head_cr1_grd, head_dl1_grd, head_dr1_grd,  head_cc1_grd, head_dc1_grd,
            head_cl2_grd, head_cr2_grd, head_dl2_grd, head_dr2_grd, head_cc2_grd, head_dc2_grd,
            head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd,
            root_c_grd, root_d_grd,

            alpha_l, alpha_r, alpha_ld, alpha_rd, alpha_cc, alpha_cd, alpha_dc, alpha_dd,
            alpha_tmp_c1, alpha_tmp_c2,

            B, L, r1, r2, r3,r4
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

        # assert head_cl1_grd.max() <= 0
        # assert head_cr1_grd.max() <= 0
        # assert head_cc1_grd.max() <= 0
        # assert head_dc1_grd.max() <= 0
        # assert head_cd1_grd.max() <= 0
        # assert head_dd1_grd.max() <= 0
        # pdb.set_trace()


        return  head_cl1_grd, head_cr1_grd, head_dl1_grd, head_dr1_grd, head_cc1_grd, head_dc1_grd, \
                head_cl2_grd, head_cr2_grd, head_dl2_grd, head_dr2_grd, head_cc2_grd, head_dc2_grd, \
                head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd, \
                alpha_l[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_r[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_ld[:, torch.arange(L - 1) + 1, torch.arange(L - 1)], \
                alpha_rd[:, torch.arange(L - 1) + 1, torch.arange(L - 1)], \
                alpha_cc[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_dc[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                root_c_grd, root_d_grd, None

class LCFRS():
    def __init__(self):
        super(LCFRS, self).__init__()

    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        head_dd1 = rules['head_dd1'].contiguous()
        head_cd1 = rules['head_cd1'].contiguous()
        head_dd2 = rules['head_dd2'].contiguous()
        head_cd2 = rules['head_cd2'].contiguous()

        unary_l = rules['unary_l'].contiguous()
        unary_ld = rules['unary_ld'].contiguous()
        unary_r = rules['unary_r'].contiguous()
        unary_rd = rules['unary_rd'].contiguous()
        unary_c = rules['unary_c'].contiguous()
        unary_d = rules['unary_d'].contiguous()

        head_cl1 = rules['head_cl1'].contiguous()
        head_cl2 = rules['head_cl2'].contiguous()
        head_dl1 = rules['head_dl1'].contiguous()
        head_dl2 = rules['head_dl2'].contiguous()
        head_cr1 = rules['head_cr1'].contiguous()
        head_cr2 = rules['head_cr2'].contiguous()
        head_dr1 = rules['head_dr1'].contiguous()
        head_dr2 = rules['head_dr2'].contiguous()

        head_cc1 = rules['head_cc1'].contiguous()
        head_cc2 = rules['head_cc2'].contiguous()
        head_dc1 = rules['head_dc1'].contiguous()
        head_dc2 = rules['head_dc2'].contiguous()

        root_c = rules['root_c'].contiguous()
        root_d = rules['root_d'].contiguous()


        return {'partition': InsideCPD_rankspace.apply(head_cl1, head_cr1,
                                                       head_dl1, head_dr1,
                                                       head_cc1, head_dc1,
                     head_cl2, head_cr2, head_dl2, head_dr2,  head_cc2, head_dc2,
                     head_cd1, head_cd2, head_dd1, head_dd2,
                     unary_l, unary_r, unary_ld, unary_rd, unary_c, unary_d,
                     root_c, root_d, lens)}


    def compute_marginals(self, rules, lens, mbr=True):
        head_dd1 = rules['head_dd1'].contiguous()
        head_cd1 = rules['head_cd1'].contiguous()
        head_dd2 = rules['head_dd2'].contiguous()
        head_cd2 = rules['head_cd2'].contiguous()

        unary_l = rules['unary_l'].contiguous()
        unary_ld = rules['unary_ld'].contiguous()
        unary_r = rules['unary_r'].contiguous()
        unary_rd = rules['unary_rd'].contiguous()
        unary_c = rules['unary_c'].contiguous()
        unary_d = rules['unary_d'].contiguous()

        head_cl1 = rules['head_cl1'].contiguous()
        head_cl2 = rules['head_cl2'].contiguous()
        head_dl1 = rules['head_dl1'].contiguous()
        head_dl2 = rules['head_dl2'].contiguous()
        head_cr1 = rules['head_cr1'].contiguous()
        head_cr2 = rules['head_cr2'].contiguous()
        head_dr1 = rules['head_dr1'].contiguous()
        head_dr2 = rules['head_dr2'].contiguous()

        head_cc1 = rules['head_cc1'].contiguous()
        head_cc2 = rules['head_cc2'].contiguous()
        head_dc1 = rules['head_dc1'].contiguous()
        head_dc2 = rules['head_dc2'].contiguous()

        root_c = rules['root_c'].contiguous()
        root_d = rules['root_d'].contiguous()

        # forward
        B, L, r1 = unary_l.shape
        r2 = unary_c.shape[-1]
        r3 = unary_ld.shape[-1]
        r4 = unary_d.shape[-1]

        L+=1

        alpha_l = unary_l.new_zeros(B, L, L, r1).fill_(0).contiguous()
        alpha_r = unary_l.new_zeros(B, L, L, r1).fill_(0).contiguous()
        alpha_cc = unary_l.new_zeros(B, L, L, r2).fill_(0).contiguous()
        alpha_dc = unary_l.new_zeros(B, L, L, 4, r4).fill_(0).contiguous()

        alpha_ld = unary_l.new_zeros(B, L, L, r3).fill_(0).contiguous()
        alpha_rd = unary_l.new_zeros(B, L, L, r3).fill_(0).contiguous()

        #
        alpha_cd = unary_l.new_zeros(B, L, L, L, L, r2).fill_(0).contiguous()
        alpha_dd = unary_l.new_zeros(B, L, L, L, L, r4).fill_(0).contiguous()

        alpha_tmp_c1 = unary_l.new_zeros(B, L, L, r1).fill_(0).contiguous()
        alpha_tmp_c2 = unary_l.new_zeros(B, L, L, r2).fill_(0).contiguous()

        alpha_l[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_l
        alpha_ld[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_ld
        alpha_r[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_r
        alpha_rd[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_rd
        alpha_cc[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_c
        alpha_dc[:, torch.arange(L-1), torch.arange(L-1) + 1] = unary_d

        inside_cuda.forward(
            head_cl1, head_cr1,
            head_dl1, head_dr1,
            head_cc1, head_dc1,
            head_cl2, head_cr2,
            head_dl2, head_dr2,
            head_cc2, head_dc2,
            head_cd1, head_cd2, head_dd1, head_dd2,
            root_c, root_d,

            alpha_l, alpha_r,
            alpha_ld, alpha_rd,
            alpha_cc, alpha_cd,
            alpha_dc, alpha_dd,
            alpha_tmp_c1, alpha_tmp_c2,
            B, L, r1, r2, r3, r4
        )

        partition = alpha_l[:, 0, -1, 0].detach()

        # backward
        head_cl1_grd = torch.zeros_like(head_cl1)
        head_dl1_grd = torch.zeros_like(head_dl1)
        head_cl2_grd = torch.zeros_like(head_cl2)
        head_dl2_grd = torch.zeros_like(head_dl2)

        head_cr1_grd = torch.zeros_like(head_cr1)
        head_dr1_grd = torch.zeros_like(head_dr1)
        head_cr2_grd = torch.zeros_like(head_cr2)
        head_dr2_grd = torch.zeros_like(head_dr2)

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
            head_cl1, head_cr1,
            head_dl1, head_dr1,
            head_cc1, head_dc1,
            head_cl2, head_cr2,
            head_dl2, head_dr2,
            head_cc2, head_dc2,
            head_cd1, head_cd2,
            head_dd1, head_dd2,
            root_c, root_d,

            head_cl1_grd, head_cr1_grd, head_dl1_grd, head_dr1_grd,  head_cc1_grd, head_dc1_grd,
            head_cl2_grd, head_cr2_grd, head_dl2_grd, head_dr2_grd, head_cc2_grd, head_dc2_grd,
            head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd,
            root_c_grd, root_d_grd,

            alpha_l, alpha_r, alpha_ld, alpha_rd, alpha_cc, alpha_cd, alpha_dc, alpha_dd,
            alpha_tmp_c1, alpha_tmp_c2,

            B, L, r1, r2, r3,r4
        )

        if mbr:
            return alpha_l, alpha_r, alpha_ld, alpha_rd, alpha_cd, alpha_dd, alpha_cc, alpha_dc, partition

        else:
            return head_cl1_grd, head_cr1_grd, head_dl1_grd, head_dr1_grd, head_cc1_grd, head_dc1_grd, \
                head_cl2_grd, head_cr2_grd, head_dl2_grd, head_dr2_grd, head_cc2_grd, head_dc2_grd, \
                head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd, \
                alpha_l[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_r[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_ld[:, torch.arange(L - 1) + 1, torch.arange(L - 1)], \
                alpha_rd[:, torch.arange(L - 1) + 1, torch.arange(L - 1)], \
                alpha_cc[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_dc[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                root_c_grd, root_d_grd

    def decode(self, rules, lens, raw_word,  viterbi=False, mbr=False):
        alpha_l, alpha_r, alpha_ld, alpha_rd, alpha_cd, alpha_dd, alpha_cc, alpha_dc, partition = self.compute_marginals(rules, lens, mbr=True)
        B, L = alpha_l.shape[:2]
        marginal_c = alpha_l.transpose(1, 2).sum(-1) + alpha_r.transpose(1, 2).sum(-1) \
                    + alpha_cc.transpose(1, 2).sum(-1) + alpha_dc.transpose(1, 2).sum([-1, -2]) \
                    + alpha_ld.transpose(1,2).sum(-1) + alpha_rd.transpose(1, 2).sum(-1)

        marginal_d = alpha_cd.transpose(1,2).sum(-1) + alpha_dd.transpose(1, 2).sum(-1)

        marginal_c = marginal_c.contiguous()
        marginal_d = marginal_d.contiguous()
        prediction, predicted_trees = mbr_decoding(marginal_c, marginal_d, raw_word, lens)
        
        return {'prediction': prediction,
                'partition': partition,
                'prediction_tree': predicted_trees}
