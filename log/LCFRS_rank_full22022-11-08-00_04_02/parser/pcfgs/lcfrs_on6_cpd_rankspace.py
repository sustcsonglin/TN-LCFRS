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
inside_cuda = load(name="ugjgjg",
                   sources=["/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on6_rank/lcfrs_cpd.cpp", "/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on6_rank/lcfrs_cpd.cu"],
                   verbose=True)

from .trees import Tree, Token

class InsideCPD_rankspace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, head_cl1, head_cr1,
                     head_dl1, head_dr1,
                     head_cc1, head_dc1,
                     head_cl2, head_cr2,
                     head_dl2, head_dr2,
                     head_cc2, head_dc2,

                     head_cd,
                     head_dd,
                     head_ill_in,
                     head_ill_out,

                     unary_l, unary_r,
                     unary_ld, unary_rd,
                     unary_c, unary_d,
                     root_c, root_d, lens):

        B, L, r1 = unary_l.shape
        r2 = unary_c.shape[-1]
        r3 = unary_ld.shape[-1]
        r4 = unary_d.shape[-1]
        r5 = head_cd.shape[-1] - r4 - r3
        # pdb.set_trace()
        # print(r1,r2,r3,r4,r5)
        # print(head_cd.shape)

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
        alpha_ill_in = unary_l.new_zeros(B, L, L, L, L, r5).fill_(0).contiguous()
        alpha_ill_out = unary_l.new_zeros(B, L, L, L, L, r5).fill_(0).contiguous()

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
            head_cd,
            head_dd,
            head_ill_in,
            head_ill_out,
            root_c, root_d,
            alpha_l, alpha_r,
            alpha_ld, alpha_rd,
            alpha_cc, alpha_cd,
            alpha_dc, alpha_dd,
            alpha_ill_in, alpha_ill_out,
            alpha_tmp_c1, alpha_tmp_c2,
            B, L, r1, r2, r3, r4, r5
        )

        assert ~alpha_ill_in.isnan().all()
        assert ~alpha_ill_out.isnan().all()


        ctx.save_for_backward(
            head_cl1, head_cr1,
            head_dl1, head_dr1,
            head_cc1, head_dc1,
            head_cl2, head_cr2,
            head_dl2, head_dr2,
            head_cc2, head_dc2,
            head_cd,
            head_dd,
            head_ill_in,
            head_ill_out,
            root_c, root_d,
            alpha_l, alpha_r,
            alpha_ld, alpha_rd,
            alpha_cc, alpha_cd,
            alpha_dc, alpha_dd,
            alpha_ill_in, alpha_ill_out,
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
        head_cd,\
        head_dd,\
        head_ill_in, \
        head_ill_out, \
        root_c, root_d,\
        alpha_l, alpha_r,\
        alpha_ld, alpha_rd,\
        alpha_cc, alpha_cd,\
        alpha_dc, alpha_dd,\
        alpha_ill_in, alpha_ill_out, \
        alpha_tmp_c1, alpha_tmp_c2 = ctx.saved_tensors

        B, L = alpha_l.shape[:2]
        r1 = alpha_l.shape[-1]
        r2 = alpha_cd.shape[-1]
        r3 = alpha_ld.shape[-1]
        r4 = alpha_dd.shape[-1]
        r5 = head_cd.shape[-1] - r4 - r3

        print(r1, r2, r3, r4, r5)

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

        head_cd_grd = torch.zeros_like(head_cd)
        head_dd_grd = torch.zeros_like(head_dd)

        head_ill_in_grd = torch.zeros_like(head_ill_in)
        head_ill_out_grd = torch.zeros_like(head_ill_out)

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

            head_cd,
            head_dd,
            head_ill_in,
            head_ill_out,
            root_c, root_d,

            head_cl1_grd, head_cr1_grd, head_dl1_grd, head_dr1_grd, head_cc1_grd, head_dc1_grd,
            head_cl2_grd, head_cr2_grd, head_dl2_grd, head_dr2_grd, head_cc2_grd, head_dc2_grd,
            head_cd_grd,
            head_dd_grd,
            head_ill_in_grd,
            head_ill_out_grd,
            root_c_grd, root_d_grd,

            alpha_l, alpha_r, alpha_ld, alpha_rd, alpha_cc, alpha_cd, alpha_dc, alpha_dd,
            alpha_ill_in, alpha_ill_out,
            alpha_tmp_c1, alpha_tmp_c2,
            B, L, r1, r2, r3, r4, r5
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

        # assert head_cl1_grd.max() <= 0
        # assert head_cr1_grd.max() <= 0
        # assert head_cc1_grd.max() <= 0
        # assert head_dc1_grd.max() <= 0
        # assert head_cd1_grd.max() <= 0
        # assert head_dd1_grd.max() <= 0
        # pdb.set_trace()

        return  head_cl1_grd, head_cr1_grd, head_dl1_grd, head_dr1_grd, head_cc1_grd, head_dc1_grd, \
                head_cl2_grd, head_cr2_grd, head_dl2_grd, head_dr2_grd, head_cc2_grd, head_dc2_grd, \
                head_cd_grd, \
                head_dd_grd, \
                head_ill_in_grd, \
                head_ill_out_grd, \
                alpha_l[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_r[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_ld[:, torch.arange(L - 1) + 1, torch.arange(L - 1)], \
                alpha_rd[:, torch.arange(L - 1) + 1, torch.arange(L - 1)], \
                alpha_cc[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_dc[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                root_c_grd, root_d_grd, None

class PCFG(PCFG_base):
    def __init__(self):
        super(PCFG, self).__init__()

    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        head_cd = rules['head_cd'].contiguous()
        head_dd = rules['head_dd'].contiguous()
        head_ill_in = rules['head_ill_in'].contiguous()
        head_ill_out = rules['head_ill_out'].contiguous()


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
                     head_cl2, head_cr2,
                     head_dl2, head_dr2,
                     head_cc2, head_dc2,

                     head_cd,
                     head_dd,
                     head_ill_in,
                     head_ill_out,

                     unary_l, unary_r,
                     unary_ld, unary_rd,
                     unary_c, unary_d,
                     root_c, root_d, lens)}


    def compute_marginals(self, rules, lens, mbr=True):
        head_dd = rules['head_dd'].contiguous()
        head_cd = rules['head_cd'].contiguous()

        head_ill_in = rules['head_ill_in'].contiguous()
        head_ill_out = rules['head_ill_out'].contiguous()

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
        r5 = head_ill_in.shape[-1] - r3 - r4

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
        alpha_ill_in = unary_l.new_zeros(B, L, L, L, L, r5).fill_(0).contiguous()
        alpha_ill_out = unary_l.new_zeros(B, L, L, L, L, r5).fill_(0).contiguous()

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
            head_cd,
            head_dd,
            head_ill_in,
            head_ill_out,
            root_c, root_d,
            alpha_l, alpha_r,
            alpha_ld, alpha_rd,
            alpha_cc, alpha_cd,
            alpha_dc, alpha_dd,
            alpha_ill_in, alpha_ill_out,
            alpha_tmp_c1, alpha_tmp_c2,
            B, L, r1, r2, r3, r4, r5
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

        head_cd_grd = torch.zeros_like(head_cd)
        head_dd_grd = torch.zeros_like(head_dd)

        head_ill_in_grd = torch.zeros_like(head_ill_in)
        head_ill_out_grd = torch.zeros_like(head_ill_out)

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
            head_cd,
            head_dd,
            head_ill_in,
            head_ill_out,

            root_c, root_d,

            head_cl1_grd, head_cr1_grd, head_dl1_grd, head_dr1_grd, head_cc1_grd, head_dc1_grd,
            head_cl2_grd, head_cr2_grd, head_dl2_grd, head_dr2_grd, head_cc2_grd, head_dc2_grd,
            head_cd_grd,
            head_dd_grd,
            head_ill_in_grd,
            head_ill_out_grd,

            root_c_grd, root_d_grd,

            alpha_l, alpha_r, alpha_ld, alpha_rd, alpha_cc, alpha_cd, alpha_dc, alpha_dd,
            alpha_ill_in, alpha_ill_out,
            alpha_tmp_c1, alpha_tmp_c2,

            B, L, r1, r2, r3, r4, r5
        )

        if mbr:
            return alpha_l, alpha_r, alpha_ld, alpha_rd, alpha_cd, alpha_dd, alpha_cc, alpha_dc, alpha_ill_in, alpha_ill_out, partition

        else:
            return head_cl1_grd, head_cr1_grd, head_dl1_grd, head_dr1_grd, head_cc1_grd, head_dc1_grd, \
                head_cl2_grd, head_cr2_grd, head_dl2_grd, head_dr2_grd, head_cc2_grd, head_dc2_grd, \
                head_cd_grd, head_cd_grd, head_dd_grd, head_dd_grd, \
                alpha_l[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_r[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_ld[:, torch.arange(L - 1) + 1, torch.arange(L - 1)], \
                alpha_rd[:, torch.arange(L - 1) + 1, torch.arange(L - 1)], \
                alpha_cc[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                alpha_dc[:, torch.arange(L-1)+1, torch.arange(L-1)], \
                root_c_grd, root_d_grd



    def decode(self, rules, lens, raw_word, viterbi=False, mbr=False):
        alpha_l, alpha_r, alpha_ld, alpha_rd, alpha_cd, alpha_dd, alpha_cc, alpha_dc, alpha_ill_in, alpha_ill_out, partition = self.compute_marginals(rules, lens, mbr=True)

        B, L = alpha_l.shape[:2]
        marginal_c = alpha_l.transpose(1, 2).sum(-1) + alpha_r.transpose(1, 2).sum(-1) \
                    + alpha_cc.transpose(1, 2).sum(-1) + alpha_dc.transpose(1, 2).sum([-1, -2]) \
                    + alpha_ld.transpose(1,2).sum(-1) + alpha_rd.transpose(1, 2).sum(-1)

        marginal_d = alpha_cd.transpose(1,2).sum(-1) + alpha_dd.transpose(1, 2).sum(-1) + alpha_ill_in.transpose(1, 2).sum(-1) + alpha_ill_out.transpose(1, 2).sum(-1)

        alpha_c_mbr = torch.zeros_like(marginal_c)
        alpha_d_mbr = torch.zeros_like(marginal_d)
        # initialize the trivial single_span case.

        alpha_c_mbr[:, torch.arange(L-1), 1+torch.arange(L-1)] = 1

        inside_cuda2.argmax_on6wn(marginal_c, marginal_d, alpha_c_mbr, alpha_d_mbr, B, L)
        prediction = [[[], []] for _ in range(B)]



        def backtrack(b_idx, start, gap_start, gap_end, end, root=False):
            if start + 1  == end:
                token_node = [Token(raw_word[b_idx][start], start, 'PP')]
                if root:
                    token_node = Tree("NP", token_node)
                return token_node

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
                    child1 = backtrack(b_idx, split1, -1, -1, split2)
                    child2 = backtrack(b_idx, start, split1, split2, end)

                # 说明这个continuous span是由两个小的continuous span组成而来的.
                else:
                    split = idx
                    assert start<split<end, f"({start}, {split}, {end})"
                    prediction[b_idx][0].append((start, split))
                    prediction[b_idx][0].append((split, end))
                    child1 = backtrack(b_idx, start, -1, -1, split)
                    child2 = backtrack(b_idx, split, -1, -1, end)

            # discontinuous的span
            else:
                idx = int(alpha_d_mbr[b_idx, gap_start, start, gap_end, end])
                if idx < 0:
                    if idx == -1:
                        prediction[b_idx][0].append((start, gap_start))
                        prediction[b_idx][0].append((gap_end, end))
                        child1 = backtrack(b_idx, start, -1, -1, gap_start)
                        child2 = backtrack(b_idx, gap_end, -1, -1, end)
                    else:
                        # on6 well-nested.
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

        return {'prediction': prediction,
                'partition': partition,
                'prediction_tree': predicted_trees}
