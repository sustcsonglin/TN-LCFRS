import pdb

import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from parser.modules.charRNN import CharProbRNN
from ..pcfgs.lcfrs_on5_cpd_rankspace_full2 import PCFG

def Nothing(x):
    return x

def softmax(a, b, dim=-1):
    return ((a @ b)).softmax(dim)

class NeuralLCFRS_rankspace_ill(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralLCFRS_rankspace_ill, self).__init__()
        self.pcfg = PCFG()
        self.device = dataset.device
        self.args = args
        # self.activate =
        self.NT = args.NT
        self.T = args.T
        if self.args.activation == 'tanh':
            self.activate = nn.Tanh()
        elif self.args.activation == 'relu':
            self.activate = nn.ReLU()
        else:
            raise NotImplementedError

        self.D = args.D

        self.r = args.r1 + args.r2
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim

        self.nonterm_emb = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.d_emb = nn.Parameter(torch.randn(self.D, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.r1 =  self.args.r1
        self.r2 = self.args.r2
        self.r3 = self.args.r3
        self.r4 = self.args.r4


        self.use_char = self.args.use_char
        self.share_r = self.args.share_r
        self.sep_emb = self.args.sep_emb

        if self.args.layer==3:
            self.term_mlp = nn.Sequential(
                nn.Linear(self.s_dim, self.s_dim),
                ResLayer(self.s_dim, self.s_dim),
                # ResLayer(self.s_dim, self.s_dim),
                # ResLayer(self.s_dim, self.s_dim),
            )

        else:
            self.term_mlp = nn.Sequential(
                nn.Linear(self.s_dim, self.s_dim),
                ResLayer(self.s_dim, self.s_dim),
                ResLayer(self.s_dim, self.s_dim),
            )
        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT))

        tmp = self.s_dim

        self.decision =nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, 4))

        if not self.args.share_r:
            self.left_c = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate, nn.Linear(tmp, self.r1 ))
            self.right_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate, nn.Linear(tmp, self.r1))
            self.parent_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate, nn.Linear(tmp, self.r1))
            self.parent_c2 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate,nn.Linear(tmp, self.r2))
            self.cd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate,nn.Linear(tmp, self.r2))
            self.cc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate,nn.Linear(tmp, self.r2))
            self.parent_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate,nn.Linear(tmp, self.r3))
            self.parent_d2 = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate,nn.Linear(tmp, self.r4))
            self.left_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate,nn.Linear(tmp, self.r3))
            self.right_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate,nn.Linear(tmp, self.r3))
            self.dd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate,nn.Linear(tmp, self.r4))
            self.dc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate,nn.Linear(tmp, self.r4))
            self.dc_mlp1 = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate,nn.Linear(tmp, self.r4))
            self.dc_mlp2 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate,nn.Linear(tmp, self.r4))
            self.dc_mlp3 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate,nn.Linear(tmp, self.r4))
            self.dc_mlp4 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate,nn.Linear(tmp, self.r4))





        else:
            self.left_c = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
            self.right_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
            self.parent_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
            self.parent_c2 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
            self.parent_c3 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)

            self.cd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
            self.cc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
            self.parent_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
            self.parent_d2 = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
            self.left_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
            self.right_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
            self.dd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
            self.dc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
            self.dc_mlp1 = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
            self.dc_mlp2 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
            self.dc_mlp3 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
            self.dc_mlp4 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)

            self.cd_ill_in =  nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
            self.cd_ill_out =  nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)

        self.r1_emb = nn.Parameter(torch.randn(self.s_dim, self.r1))
        self.r2_emb = nn.Parameter(torch.randn(self.s_dim, self.r2))
        self.r3_emb = nn.Parameter(torch.randn(self.s_dim, self.r3))
        self.r4_emb = nn.Parameter(torch.randn(self.s_dim, self.r4))
        self.r5_emb = nn.Parameter(torch.randn(self.s_dim, self.r5))

        self.nonterm_emb2 = nn.Parameter(torch.randn((self.NT + self.T) * 4, self.s_dim))
        self.NT_T = self.NT + self.T

        self.NT_T_D = self.NT + self.T

        if self.args.init != 'no':
             self._initialize()

        if dataset.emb is None:
            self.vocab_emb = nn.Parameter(torch.randn(self.V, self.s_dim))
            if self.args.init != 'no':
                torch.nn.init.xavier_normal_(self.vocab_emb)
        else:
            self.vocab_emb = nn.Parameter(torch.from_numpy(dataset.emb).float())

        # torch.nn.init.normal_(self.term_emb)

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)


    def forward(self, input, evaluating=False):
        x = input['word']
        b, n = x.shape[:2]

        if self.share_r:
            r1_emb = self.r1_emb
            r2_emb = self.r2_emb
            r3_emb = self.r3_emb
            r4_emb = self.r4_emb

        d_emb = self.d_emb

        root_emb = self.root_emb
        root = self.root_mlp(root_emb).softmax(-1).expand(b, -1)
        if not self.args.share_pt:
            term_prob = (self.term_mlp(self.term_emb) @ self.vocab_emb.t()).softmax(-1)
        else:
            term_prob = softmax(self.term_mlp(self.nonterm_emb[self.NT:]), self.vocab_emb.t())


        unary = term_prob[torch.arange(self.T)[None, None], x[:, :, None]]

        if not self.share_r:
            head = torch.cat([self.parent_c(self.nonterm_emb[:self.NT]), self.parent_c2(self.nonterm_emb[:self.NT])], -1).softmax(-1)

        else:
            # head = (self.parent_c(self.nonterm_emb[:self.NT]) @ self.r1_emb).softmax(-1)
            head = (torch.cat([self.parent_c(self.nonterm_emb[:self.NT]) @ self.r1_emb,
                               self.parent_c2(self.nonterm_emb[:self.NT]) @ self.r2_emb,
                               self.parent_c3(self.nonterm_emb[:self.NT]) @ self.r5_emb,
                               ], -1)).softmax(-1)

        head_c1 = head[:, :self.r1]
        head_c2 = head[:, self.r1:self.r1+self.r2]
        head_c3 = head[:, self.r1+self.r2:]


        if not self.share_r:
            left_c = (self.left_c(self.nonterm_emb)).softmax(0)
            right_c = (self.right_c(self.nonterm_emb)).softmax(0)
            cc = (self.cc_mlp(self.nonterm_emb)).softmax(0)
            cd = (self.cd_mlp(d_emb)).softmax(0)
            left_d =  (self.left_d(self.nonterm_emb)).softmax(0)
            right_d = (self.right_d(self.nonterm_emb)).softmax(0)

        else:
            left_c = softmax(self.left_c(self.nonterm_emb),  r1_emb, 0)
            right_c = softmax(self.right_c(self.nonterm_emb), r1_emb, 0)
            cc =  softmax(self.cc_mlp(self.nonterm_emb3), r2_emb, 0)
            cd = softmax(self.cd_mlp(d_emb), r2_emb, 0)
            left_d = softmax(self.left_d(self.nonterm_emb), r3_emb, 0)
            right_d = softmax(self.right_d(self.nonterm_emb), r3_emb, 0)

            cd_in = (self.cd_ill_in(d_emb) @ self.r5_emb).softmax(0)
            cd_out = (self.cd_ill_out(d_emb) @ self.r5_emb).softmax(0)


        if not self.share_r:
            head_d = torch.cat([self.parent_d(d_emb), self.parent_d2(d_emb)], -1)

        else:
            head_d = torch.cat([self.parent_d(d_emb) @ self.r3_emb, self.parent_d2(d_emb) @ self.r4_emb], -1)

        if self.args.ban_nested:
            mask = torch.zeros_like(head_d)
            mask[:, self.r3:] = -1e9
            head_d = head_d + mask

        head_d = head_d.softmax(-1)

        if not self.args.share_r:
            #
            if not self.sep_emb:
                dc1 = self.dc_mlp1(self.nonterm_emb)
                dc2 = self.dc_mlp2(self.nonterm_emb)
                dc3 = self.dc_mlp3(self.nonterm_emb)
                dc4 = self.dc_mlp4(self.nonterm_emb)
                dc = torch.cat([dc1, dc2, dc3, dc4], dim=0).softmax(0).reshape(self.NT_T, 4, -1)

            else:
                dc1 = self.dc_mlp1(self.nonterm_emb)
                dc2 = self.dc_mlp2(self.nonterm_emb)
                dc3 = self.dc_mlp3(self.nonterm_emb)
                dc4 = self.dc_mlp4(self.nonterm_emb)
                dc = torch.stack([dc1, dc2, dc3, dc4], dim=-1)
                decision = self.decision(r4_emb.t()).softmax(-1)
                dc = torch.einsum('cr, rd -> cdr', dc, decision)

            # dc = cc
            dd = (self.dd_mlp(self.d_emb)).softmax(0)

        else:
            if not self.sep_emb:

                if self.args.merge:
                    dc1 = self.dc_mlp1(self.nonterm_emb3) @ r4_emb
                    dc2 = self.dc_mlp2(self.nonterm_emb3) @ r4_emb
                    dc3 = self.dc_mlp3(self.nonterm_emb3) @ r4_emb
                    dc4  = self.dc_mlp4(self.nonterm_emb3) @ r4_emb
                    dc = torch.cat([dc1, dc2, dc3, dc4], dim=0).softmax(0).reshape(self.NT_T, 4, -1)
                else:
                    dc = softmax(self.dc_mlp(self.nonterm_emb), r4_emb, 0)
                    # dc2 = self.dc_mlp2(self.nonterm_emb) @ r4_emb
                    # dc3 = self.dc_mlp3(self.nonterm_emb) @ r4_emb
                    # dc4 = self.dc_mlp4(self.nonterm_emb) @ r4_emb
                    # dc = torch.stack([dc1.softmax(0), dc2.softmax(0), dc3.softmax(0), dc4.softmax(0)], dim=-1)
                    decision = self.decision(r4_emb.t()).softmax(-1)
                    dc = torch.einsum('cr, rd -> cdr', dc, decision)
                    dd = softmax(self.dd_mlp(self.d_emb), r4_emb, 0)

            else:
                if self.args.merge:
                    dc1 = self.dc_mlp1(self.nonterm_emb3)
                    dc2 = self.dc_mlp2(self.nonterm_emb3)
                    dc3 = self.dc_mlp3(self.nonterm_emb3)
                    dc4  = self.dc_mlp4(self.nonterm_emb3)
                    dc = torch.cat([dc1, dc2, dc3, dc4], dim=0).softmax(0).reshape(self.NT_T, 4, -1)

                else:

                    decision = self.decision(r4_emb.t()).softmax(-1)
                    dc = torch.einsum('cr, rd -> cdr', dc, decision)

                dd = (self.dd_mlp(self.d_emb)).softmax(0)

        head_d1 = head_d[:, :self.r3].contiguous()
        head_d2 = head_d[:, self.r3:].contiguous()

        # dd = cd
        head_cd1 = (torch.einsum('xi, xj -> ji', head_d1, cd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()
        head_cd2 = (torch.einsum('xi, xj -> ji', head_d2, cd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()

        head_cd_in1 = (torch.einsum('xi, xj -> ji', head_d1, cd_in) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()
        head_cd_in2 = (torch.einsum('xi, xj -> ji', head_d2, cd_in) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()

        head_cd_out1 = (torch.einsum('xi, xj -> ji', head_d1, cd_out) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()
        head_cd_out2 = (torch.einsum('xi, xj -> ji', head_d2, cd_out) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()


        head_dd1 = (torch.einsum('xi, xj -> ji', head_d1, dd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()
        head_dd2 = (torch.einsum('xi, xj -> ji', head_d2, dd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()



        unary_l = (torch.einsum('blp, pr -> blr', unary, left_c[..., self.NT:, :]) + 1e-9).log()
        unary_ld = (torch.einsum('blp, pr -> blr', unary, left_d[..., self.NT:, :]) + 1e-9).log()
        unary_r = (torch.einsum('blp, pr -> blr', unary, right_c[..., self.NT:, :]) + 1e-9).log()
        unary_rd = (torch.einsum('blp, pr -> blr', unary, right_d[..., self.NT:, :]) + 1e-9).log()

        unary_c = (torch.einsum('blp, pr -> blr', unary, cc[..., self.NT:, :]) + 1e-9).log()
        unary_d = (torch.einsum('blp, pdr -> bldr', unary, dc[self.NT:, :, :]) + 1e-9).log()

        head_cl1 = (torch.einsum('xi, xj -> ji', head_c1, left_c[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b,
                                                                                                                      1,
                                                                                                                     1).contiguous()
        head_cl2 = (torch.einsum('xi, xj -> ji', head_c2, left_c[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b,1,                            1,
                                                                                                                      1).contiguous()

        head_cl3 = (torch.einsum('xi, xj -> ji', head_c3, left_c[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b,1,                            1,
                                                                                                                      1).contiguous()


        head_dl1 = (torch.einsum('xi, xj -> ji', head_c1, left_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()
        head_dr1 = (torch.einsum('xi, xj -> ji', head_c1, right_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()

        head_dl2 = (torch.einsum('xi, xj -> ji', head_c2, left_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()
        head_dr2 = (torch.einsum('xi, xj -> ji', head_c2, right_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()

        head_dl3 = (torch.einsum('xi, xj -> ji', head_c3, left_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()
        head_dr3 = (torch.einsum('xi, xj -> ji', head_c3, right_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()


        head_cc1 = (torch.einsum('xi, xj -> ji', head_c1, cc[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,
                                                                                                                  1).contiguous()
        head_cc2 = (torch.einsum('xi, xj -> ji', head_c2, cc[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,
                                                                                                                  1).contiguous()
        head_cc3 = (torch.einsum('xi, xj -> ji', head_c2, cc[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,
                                                                                                                  1).contiguous()

        head_dc1 = (torch.einsum('xi, xdj -> jdi', head_c1, dc[..., :self.NT, :, :]) + 1e-9).log().unsqueeze(0).repeat(
            b, 1, 1, 1).contiguous()
        head_dc2 = (torch.einsum('xi, xdj -> jdi', head_c2, dc[:self.NT, :, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,
                                                                                                                  1,
                                                                                                                  1).contiguous()
        head_dc3 = (torch.einsum('xi, xdj -> jdi', head_c3, dc[..., :self.NT, :, :]) + 1e-9).log().unsqueeze(0).repeat(
            b, 1, 1, 1).contiguous()

        head_cr1 = (torch.einsum('xi, xj -> ji', head_c1, right_c[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(
            b, 1, 1).contiguous()
        head_cr2 = (torch.einsum('xi, xj -> ji', head_c2, right_c[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(
            b, 1, 1).contiguous()
        head_cr3 = (torch.einsum('xi, xj -> ji', head_c3, right_c[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(
            b, 1, 1).contiguous()

        root_c = (torch.einsum('bm, mr -> br', root, head_c1) + 1e-9).log()
        root_d = (torch.einsum('bm, mr -> br', root, head_c2) + 1e-9).log()

        return {'head_dd1': head_dd1,
                'head_cd1': head_cd1,
                'head_dd2': head_dd2,
                'head_cd2': head_cd2,

                'head_cd_in1': head_cd_in1,
                'head_cd_in2': head_cd_in2,
                'head_cd_out1': head_cd_out1,
                'head_cd_out2': head_cd_out2,

                'unary_l': unary_l,
                'unary_ld': unary_ld,
                'unary_r': unary_r,
                'unary_rd': unary_rd,
                'unary_c': unary_c,
                'unary_d': unary_d,

                'head_cl1': head_cl1,
                'head_cl2': head_cl2,
                'head_cl3': head_cl3,

                'head_cr1': head_cr1,
                'head_cr2': head_cr2,
                'head_cr3': head_cr3,

                'head_dl1': head_dl1,
                'head_dl2': head_dl2,
                'head_dl3': head_dl3,

                'head_dr1': head_dr1,
                'head_dr2': head_dr2,
                'head_dr3': head_dr3,

                'head_cc1': head_cc1,
                'head_cc2': head_cc2,
                'head_cc3': head_cc3,

                'head_dc1': head_dc1,
                'head_dc2': head_dc2,
                'head_dc3': head_dc3,

                'root_c': root_c,
                'root_d': root_d,

                'kl': torch.tensor(0, device=self.device)}



    def loss(self, input):
        rules = self.forward(input)
        result = self.pcfg._inside(rules=rules, lens=input['seq_len'])
        return -result['partition'].mean()

    def loss_em(self, input, model2):
        rules = self.forward(input)

        with torch.no_grad():
            rules2 = model2.forward(input)

            head_cl1_grd, head_cr1_grd, head_dl1_grd, head_dr1_grd, head_cc1_grd, head_dc1_grd, \
            head_cl2_grd, head_cr2_grd, head_dl2_grd, head_dr2_grd, head_cc2_grd, head_dc2_grd, \
            head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd, \
            unary_l_grd, \
            unary_r_grd, \
            unary_ld_grd, \
            unary_rd_grd, \
            unary_cc_grd, \
            unary_dc_grd, \
            root_c_grd, root_d_grd \
            = self.pcfg.compute_marginals(
                rules2, input['seq_len'], False)

        head_dl1 = rules['head_dl1']
        head_dl2 = rules['head_dl2']
        head_cr1 = rules['head_cr1']
        head_cr2 = rules['head_cr2']
        head_dr1 = rules['head_dr1']
        head_dr2 = rules['head_dr2']

        head_cc1 = rules['head_cc1']
        head_cc2 = rules['head_cc2']
        head_dc1 = rules['head_dc1']
        head_dc2 = rules['head_dc2']

        root_c = rules['root_c']
        root_d = rules['root_d']

        loss = (rules['head_dd1'] * head_dd1_grd).sum() + (rules['head_cd1'] * head_cd1_grd).sum() + \
        (rules['head_dd2'] * head_dd2_grd).sum() + (rules['head_cd2'] * head_cd2_grd).sum() + \
        (rules['unary_l'] * unary_l_grd).sum() + (rules['unary_ld'] * unary_ld_grd).sum() + \
        (rules['unary_r'] * unary_r_grd).sum() + (rules['unary_rd'] * unary_rd_grd).sum() + \
        (rules['unary_c'] * unary_cc_grd).sum() + (rules['unary_d'] * unary_dc_grd).sum() + \
        (rules['head_cl1'] * head_cl1_grd).sum() + (rules['head_cl2'] * head_cl2_grd).sum() + \
        (head_dl1 * head_dl1_grd).sum() + (head_dl2 * head_dl2_grd).sum() + \
        (head_cr1 * head_cr1_grd).sum() + (head_cr2 * head_cr2_grd).sum() + \
        (head_dr1 * head_dr1_grd).sum() + (head_dr2 * head_dr2_grd).sum() + \
        (head_cc1 * head_cc1_grd).sum() + (head_cc2 * head_cc2_grd).sum() + \
        (head_dc1 * head_dc1_grd).sum() + (head_dc2 * head_dc2_grd).sum() + \
        (root_c * root_c_grd).sum() + (root_d * root_d_grd).sum()

        loss = -loss / head_cc1_grd.shape[0]
        return loss

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'],  viterbi=True, mbr=False, raw_word=input['raw_word'])
        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True, raw_word=input['raw_word'])
        else:

            raise NotImplementedError



