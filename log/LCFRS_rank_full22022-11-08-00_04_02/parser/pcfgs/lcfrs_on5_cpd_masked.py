from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import stripe, diagonal_copy_, diagonal, checkpoint
import torch
import os
import numpy as np
from torch.utils.cpp_extension import load
#

try:
    inside_cuda = load(name="inside_on5_cpd_mask",
                   sources=["/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_mask/lcfrs_cpd.cpp", "/public/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_mask/lcfrs_cpd.cu"],
                   verbose=True)

except:
    inside_cuda = load(name="inside_on5_cpd",
                       sources=["/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_mask/lcfrs_cpd.cpp", "/home/yangsl/code/TN-PCFG-main/parser/pcfgs/lcfrs_on5_cpd_mask/lcfrs_cpd.cu"],
                       verbose=True)

class InsideCPD_masked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, head_c1, head_c2, head_d1, head_d2, left_c, right_c, left_d, right_d,
                cc, cd, dc, dd, unary, root, lens, alpha_mask, alpha_d_mask):
        B, L, p = unary.shape
        m = left_c.shape[1] - p
        r1 = head_c1.shape[-1]
        r2 = head_c2.shape[-1]
        r3 = head_d1.shape[-1]
        r4 = head_d2.shape[-1]
        d = dd.shape[1]
        L+=1
        alpha = unary.new_zeros(B, L, L, m).fill_(-1e9)
        alpha_d = unary.new_zeros(B, L, L, L, L, d).fill_(-1e9)

        # 看显存大小吧..
        alpha_cd = unary.new_zeros(B, L, L, L, L, r2).fill_(-1e9)
        alpha_dd = unary.new_zeros(B, L, L, L, L, r4).fill_(-1e9)

        alpha_lc = unary.new_zeros(B, L, L, r1).fill_(-1e9)
        alpha_rc = unary.new_zeros(B, L, L, r1).fill_(-1e9)
        alpha_ld = unary.new_zeros(B, L, L, r3).fill_(-1e9)
        alpha_rd = unary.new_zeros(B, L, L, r3).fill_(-1e9)
        alpha_cc = unary.new_zeros(B, L, L, r2).fill_(-1e9)
        alpha_dc = unary.new_zeros(B, L, L, 4, r4).fill_(-1e9)

        inside_cuda.forward(head_c1, head_c2, head_d1, head_d2, left_c, right_c, left_d, right_d,
                cc, cd, dc, dd, unary, alpha,  alpha_d, alpha_mask, alpha_d_mask,
                alpha_lc, alpha_rc, alpha_ld, alpha_rd,
                alpha_cc, alpha_cd, alpha_dc, alpha_dd,
                B, L, m, p, d, r1, r2, r3, r4)

        to_return1 = (alpha[torch.arange(B), 0, lens] + root)
        to_return2 = to_return1.logsumexp(-1)
        # print(alpha)

        ctx.save_for_backward(head_c1, head_c2, head_d1, head_d2, left_c, right_c, left_d, right_d,
                cc, cd, dc, dd, unary, lens, alpha,  alpha_d, alpha_mask, alpha_d_mask,
                  alpha_lc, alpha_rc, alpha_ld, alpha_rd, alpha_cc, alpha_cd, alpha_dc, alpha_dd, to_return1, to_return2)

        return to_return2

    @staticmethod
    def backward(ctx, gwk):
        head_c1, head_c2, head_d1, head_d2, left_c, right_c, left_d, right_d, \
        cc, cd, dc, dd, unary, lens, alpha, alpha_d, alpha_mask, alpha_d_mask, alpha_lc, alpha_rc, alpha_ld, alpha_rd, \
        alpha_cc, alpha_cd, alpha_dc, alpha_dd, t1, t2 = ctx.saved_tensors

        alpha[alpha<-1e8] = 0
        alpha_d[alpha_d<-1e8] = 0
        alpha_cc[alpha_cc<-1e8] = 0
        alpha_lc[alpha_lc<-1e8] = 0
        alpha_ld[alpha_ld<-1e8] = 0
        alpha_rc[alpha_rc<-1e8] = 0
        alpha_rd[alpha_rd<-1e8] = 0
        alpha_cd[alpha_cd<-1e8] = 0
        alpha_dc[alpha_dc<-1e8] = 0
        alpha_dd[alpha_dd<-1e8] = 0

        head_c1_grd = torch.zeros_like(head_c1)
        head_c2_grd = torch.zeros_like(head_c2)
        head_d1_grd = torch.zeros_like(head_d1)
        head_d2_grd = torch.zeros_like(head_d2)
        left_c_grd = torch.zeros_like(left_c)
        left_d_grd = torch.zeros_like(left_d)
        right_c_grd = torch.zeros_like(right_c)
        right_d_grd = torch.zeros_like(right_d)
        cc_grd = torch.zeros_like(cc)
        cd_grd = torch.zeros_like(cd)
        dc_grd = torch.zeros_like(dc)
        dd_grd = torch.zeros_like(dd)
        unary_grd = torch.zeros_like(unary)

        gradient_root = (t1-t2.unsqueeze(-1)).exp() * gwk.unsqueeze(-1)
        batch_size = alpha.shape[0]
        alpha[torch.arange(batch_size), lens, 0] = gradient_root
        B, L, p = unary.shape
        m = left_c.shape[1] - p
        r1 = head_c1.shape[-1]
        r2 = head_c2.shape[-1]
        r3 = head_d1.shape[-1]
        r4 = head_d2.shape[-1]
        d = dd.shape[1]
        L+=1
        inside_cuda.backward(head_c1, head_c2, head_d1, head_d2, left_c, right_c, left_d, right_d,
                cc, cd, dc, dd, unary,
        head_c1_grd, head_c2_grd, head_d1_grd, head_d2_grd, left_c_grd, right_c_grd, left_d_grd, right_d_grd,
        cc_grd, cd_grd, dc_grd, dd_grd, unary_grd,
        alpha,  alpha_d, alpha_mask, alpha_d_mask,
        alpha_lc, alpha_rc, alpha_ld, alpha_rd,
        alpha_cc, alpha_cd, alpha_dc, alpha_dd,
        B, L, m, p, d, r1, r2, r3, r4)


        # print(alpha.transpose(1,2)[torch.ones(L, L).triu()[None, :, :].expand(batch, L, L).bool().cuda()].sum())
        # print(alpha_d.transpose(1,2)[torch.ones(L, L).triu()[None, :, :, None, None].expand(batch, L, L, L, L).bool().cuda()].sum())
        # alpha = alpha.transpose(1, 2)
        # alpha_d = alpha_d.transpose(1, 2)

        return head_c1_grd, head_c2_grd, head_d1_grd, head_d2_grd, left_c_grd, right_c_grd, left_d_grd, right_d_grd, \
        cc_grd, cd_grd, dc_grd, dd_grd, unary_grd, gradient_root, None, None, None



from .lcfrs_on5_cpd import PCFG

class PCFG_mask(PCFG_base):
    def __init__(self):
        super(PCFG_mask, self).__init__()

    @torch.enable_grad()
    def _inside(self, rules, lens, alpha_mask, alpha_d_mask, viterbi=False, mbr=False):

        unary = rules['unary']

        head_c1 = rules['head_c1']
        head_d1 = rules['head_d1']
        head_c2 = rules['head_c2']
        head_d2 = rules['head_d2']
        left_c = rules['left_c']
        left_d = rules['left_d']
        right_d = rules['right_d']
        right_c = rules['right_c']
        cc = rules['cc']
        cd = rules['cd']
        dc = rules['dc']
        dd = rules['dd']
        root = rules['root']

        return {'partition': InsideCPD_masked.apply(head_c1, head_c2, head_d1, head_d2, left_c, right_c, left_d, right_d,
                cc, cd, dc, dd, unary, root, lens, alpha_mask, alpha_d_mask)}


#
#