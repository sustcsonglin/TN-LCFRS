import pdb

import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from parser.modules.charRNN import CharProbRNN
from ..pcfgs.lcfrs_on5_cpd_rankspace_full4 import PCFG

def Nothing(x):
    return x

def softmax(a, b, dim=-1):
    return ((a @ b)).softmax(dim)

class NeuralLCFRS_rankspace_full4(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralLCFRS_rankspace_full4, self).__init__()
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
        self.nonterm_emb3 = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))
        self.nonterm_emb4 = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))
        self.nonterm_emb5 = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))
        self.nonterm_emb_d = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.d_emb = nn.Parameter(torch.randn(self.D, self.s_dim))
        self.d_emb2 = nn.Parameter(torch.randn(self.D, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.r1 =  self.args.r1
        self.r2 = self.args.r2
        self.r3 = self.args.r3
        self.r4 = self.args.r4

        assert self.r4 <= self.r2

        self.use_char = self.args.use_char
        self.share_r = self.args.share_r
        self.sep_emb = self.args.sep_emb

        if self.args.layer==3:
            self.term_mlp = nn.Sequential(
                nn.Linear(self.s_dim, self.s_dim),
                ResLayer(self.s_dim, self.s_dim),
                ResLayer(self.s_dim, self.s_dim),
                ResLayer(self.s_dim, self.s_dim),
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
            if self.args.use_activation:
                self.left_c = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
                self.right_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
                self.parent_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
                self.parent_c2 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
                self.cd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
                self.cc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
                self.parent_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
                self.parent_d21 = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
                self.parent_d22 = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
                self.parent_d23 = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
                self.parent_d24 = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)

                self.left_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
                self.right_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
                self.dd1_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
                self.dd2_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
                self.dd3_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
                self.dd4_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
                self.dc1_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
                self.dc2_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
                self.dc3_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
                self.dc4_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
            else:
                self.left_c = nn.Linear(self.s_dim, tmp)
                self.right_c = nn.Linear(self.s_dim, tmp)
                self.parent_c = nn.Linear(self.s_dim, tmp)
                self.parent_c2 = nn.Linear(self.s_dim, tmp)
                self.cd_mlp = nn.Linear(self.s_dim, tmp)
                self.cc_mlp = nn.Linear(self.s_dim, tmp)
                self.parent_d = nn.Linear(self.s_dim, tmp)
                self.parent_d21 = nn.Linear(self.s_dim, tmp)
                self.parent_d22 = nn.Linear(self.s_dim, tmp)
                self.parent_d23 = nn.Linear(self.s_dim, tmp)
                self.parent_d24 = nn.Linear(self.s_dim, tmp)

                self.left_d = nn.Linear(self.s_dim, tmp)
                self.right_d = nn.Linear(self.s_dim, tmp)
                self.dd1_mlp = nn.Linear(self.s_dim, tmp)
                self.dd2_mlp = nn.Linear(self.s_dim, tmp)
                self.dd3_mlp = nn.Linear(self.s_dim, tmp)
                self.dd4_mlp = nn.Linear(self.s_dim, tmp)
                self.dc1_mlp = nn.Linear(self.s_dim, tmp)
                self.dc2_mlp = nn.Linear(self.s_dim, tmp)
                self.dc3_mlp = nn.Linear(self.s_dim, tmp)
                self.dc4_mlp = nn.Linear(self.s_dim, tmp)


        self.r1_emb = nn.Parameter(torch.randn(self.s_dim, self.r1))
        self.r2_emb = nn.Parameter(torch.randn(self.s_dim, self.r2))
        self.r3_emb = nn.Parameter(torch.randn(self.s_dim, self.r3))
        self.r41_emb = nn.Parameter(torch.randn(self.s_dim, self.r4))
        self.r42_emb = nn.Parameter(torch.randn(self.s_dim, self.r4))
        self.r43_emb = nn.Parameter(torch.randn(self.s_dim, self.r4))
        self.r44_emb = nn.Parameter(torch.randn(self.s_dim, self.r4))

        self.NT_T = self.NT + self.T

        self.NT_T_D = self.NT + self.T

        self._initialize()

        if dataset.emb is None:
            self.vocab_emb = nn.Parameter(torch.randn(self.V, self.s_dim))
            if self.args.init == 'xn':
                torch.nn.init.xavier_normal_(self.vocab_emb)

            elif self.args.init == 'xu':
                torch.nn.init.xavier_uniform_(self.vocab_emb)

            elif self.args.init == 'n':
                torch.nn.init.normal_(self.vocab_emb)

            elif self.args.init == 'u':
                torch.nn.init.uniform_(self.vocab_emb)

            else:
                raise NotImplementedError


        else:
            self.vocab_emb = nn.Parameter(torch.from_numpy(dataset.emb).float())


        # torch.nn.init.normal_(self.term_emb)

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                if self.args.init == 'xn':
                    torch.nn.init.xavier_normal_(p)

                elif self.args.init == 'xu':
                    torch.nn.init.xavier_uniform_(p)

                elif self.args.init == 'n':
                    torch.nn.init.normal_(p)

                elif self.args.init == 'u':
                    torch.nn.init.uniform_(p)

                else:
                    raise NotImplementedError

    def forward(self, input, evaluating=False):
        x = input['word']
        b, n = x.shape[:2]

        r1_emb = self.r1_emb
        r2_emb = self.r2_emb
        r3_emb = self.r3_emb

        d_emb = self.d_emb

        root_emb = self.root_emb

        root = self.root_mlp(root_emb).softmax(-1).expand(b, -1)

        if not self.args.share_pt:
            term_prob = (self.term_mlp(self.term_emb) @ self.vocab_emb.t()).softmax(-1)
        else:
            term_prob = softmax(self.term_mlp(self.nonterm_emb[self.NT:]), self.vocab_emb.t())

        unary = term_prob[torch.arange(self.T)[None, None], x[:, :, None]]

        head = (torch.cat([self.parent_c(self.nonterm_emb[:self.NT]) @ self.r1_emb, self.parent_c2(self.nonterm_emb[:self.NT]) @ self.r2_emb], -1)).softmax(-1)

        head_c1 = head[:, :self.r1]
        head_c2 = head[:, self.r1:]

        left_c = softmax(self.left_c(self.nonterm_emb),  r1_emb, 0)
        right_c = softmax(self.right_c(self.nonterm_emb), r1_emb, 0)
        cc =  softmax(self.cc_mlp(self.nonterm_emb), r2_emb, 0)
        cd = softmax(self.cd_mlp(d_emb), r2_emb, 0)
        left_d = softmax(self.left_d(self.nonterm_emb), r3_emb, 0)
        right_d = softmax(self.right_d(self.nonterm_emb), r3_emb, 0)

        if self.args.type != 3:
            head_d = torch.cat([
            self.parent_d(d_emb) @ self.r3_emb,
            self.parent_d21(d_emb) @ self.r41_emb,
            # self.parent_d22(d_emb) @ self.r42_emb,
            # self.parent_d23(d_emb) @ self.r43_emb,
            # self.parent_d24(d_emb) @ self.r44_emb,
            ], -1).softmax(-1)

            head_d1 = head_d[:, :self.r3].contiguous()
            head_d2 = head_d[:, self.r3:].unsqueeze(1).expand(-1, 4,-1)

        else:
            head_d = torch.cat([
            self.parent_d(d_emb) @ self.r3_emb,
            self.parent_d21(d_emb) @ self.r41_emb,
            self.parent_d22(d_emb) @ self.r42_emb,
            self.parent_d23(d_emb) @ self.r43_emb,
            self.parent_d24(d_emb) @ self.r44_emb,
            ], -1)

            # head_d[:, 2] = -1e9
            # head_d[:, 4] = -1e9
            # head_d[:, 5] = -1e9

            head_d = head_d.softmax(-1)
            head_d1 = head_d[:, :self.r3].contiguous()
            head_d2 = head_d[:, self.r3:].reshape(self.D, 4, -1)

        if self.args.type == 1:
            dc =  (self.dc1_mlp(self.nonterm_emb) @ self.r41_emb).softmax(0).unsqueeze(1).expand(-1, 4, -1)
            dd =  torch.cat([
                    self.dd1_mlp(self.d_emb) @ self.r41_emb,
                    self.dd2_mlp(self.d_emb) @ self.r41_emb,
                    self.dd3_mlp(self.d_emb) @ self.r41_emb,
                    self.dd4_mlp(self.d_emb) @ self.r41_emb,
                    ], 0).softmax(0).reshape(4, self.D, -1).transpose(0, 1)

        elif self.args.type == 2:
            # dc = (self.dc1_mlp(self.nonterm_emb2) @ self.r41_emb).softmax(0).reshape( self.NT_T, 4, -1)
            dc =  torch.cat([
                    self.dc1_mlp(self.nonterm_emb) @ self.r41_emb,
                    self.dc2_mlp(self.nonterm_emb) @ self.r41_emb,
                    self.dc3_mlp(self.nonterm_emb) @ self.r41_emb,
                    self.dc4_mlp(self.nonterm_emb) @ self.r41_emb,
                    ], 0).softmax(0).reshape(4, self.NT_T, -1).transpose(0, 1)

            # dd =  torch.cat([
            #         self.dd1_mlp(self.d_emb) @ self.r41_emb,
            #         self.dd2_mlp(self.d_emb) @ self.r41_emb,
            #         self.dd3_mlp(self.d_emb) @ self.r41_emb,
            #         self.dd4_mlp(self.d_emb) @ self.r41_emb,
            #         ], 0).softmax(0).reshape(4, self.D, -1).transpose(0, 1)
            dd = (self.cd_mlp(self.d_emb) @ self.r41_emb).softmax(0).unsqueeze(1).expand(-1, 4, -1)

        elif self.args.type == 3:
            dc = torch.stack([
                self.dc1_mlp(self.nonterm_emb) @ self.r41_emb,
                self.dc2_mlp(self.nonterm_emb) @ self.r42_emb,
                self.dc3_mlp(self.nonterm_emb) @ self.r43_emb,
                self.dc4_mlp(self.nonterm_emb) @ self.r44_emb,
            ], -2).softmax(0)

            dd = torch.stack([
                self.dd1_mlp(self.d_emb) @ self.r41_emb,
                self.dd2_mlp(self.d_emb) @ self.r42_emb,
                self.dd3_mlp(self.d_emb) @ self.r43_emb,
                self.dd4_mlp(self.d_emb) @ self.r44_emb,
            ], -2).softmax(0)

        elif self.args.type == 4:
            dc = (self.dc1_mlp(self.nonterm_emb) @ self.r41_emb).softmax(0)
            decision = self.decision(self.r41_emb.t()).softmax(-1)
            dc = dc.unsqueeze(-2) * decision.t().unsqueeze(0)
            dd = (self.dd1_mlp(self.d_emb) @ self.r41_emb).softmax(0).unsqueeze(1).expand(-1, 4, -1)

        else:
            raise NotImplementedError



        # dd = cd
        head_cd1 = (torch.einsum('xi, xj -> ji', head_d1, cd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()
        head_cd2 = (torch.einsum('xai, xj -> jai', head_d2, cd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1,  1).contiguous()
        head_dd1 = (torch.einsum('xi, xbj -> jbi', head_d1, dd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1, 1).contiguous()
        head_dd2 = (torch.einsum('xai, xbj -> jbai', head_d2, dd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1, 1, 1).contiguous()

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

        head_dl1 = (torch.einsum('xi, xj -> ji', head_c1, left_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()
        head_dr1 = (torch.einsum('xi, xj -> ji', head_c1, right_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()

        head_dl2 = (torch.einsum('xi, xj -> ji', head_c2, left_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()
        head_dr2 = (torch.einsum('xi, xj -> ji', head_c2, right_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()


        head_cc1 = (torch.einsum('xi, xj -> ji', head_c1, cc[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,
                                                                                                                  1).contiguous()
        head_cc2 = (torch.einsum('xi, xj -> ji', head_c2, cc[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,
                                                                                                                  1).contiguous()

        head_dc1 = (torch.einsum('xi, xdj -> jdi', head_c1, dc[..., :self.NT, :, :]) + 1e-9).log().unsqueeze(0).repeat(
            b, 1, 1, 1).contiguous()
        head_dc2 = (torch.einsum('xi, xdj -> jdi', head_c2, dc[:self.NT, :, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,
                                                                                                                  1,
                                                                                                                  1).contiguous()
        head_cr1 = (torch.einsum('xi, xj -> ji', head_c1, right_c[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(
            b, 1, 1).contiguous()
        head_cr2 = (torch.einsum('xi, xj -> ji', head_c2, right_c[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(
            b, 1, 1).contiguous()

        root_c = (torch.einsum('bm, mr -> br', root, head_c1) + 1e-9).log()
        root_d = (torch.einsum('bm, mr -> br', root, head_c2) + 1e-9).log()
        # assert ~torch.isnan(head_dd1).any()
        # assert ~torch.isnan(head_cd1).any()
        # assert ~torch.isnan(head_dd2).any()
        # assert ~torch.isnan(head_cd2).any()
        # assert ~torch.isnan(unary_l).any()
        # assert ~torch.isnan(unary_r).any()
        # assert ~torch.isnan(unary_c).any()
        # assert ~torch.isnan(unary_d).any()
        # assert ~torch.isnan(head_cl1).any()
        # assert ~torch.isnan(head_cl2).any()
        # assert ~torch.isnan(head_cr1).any()
        # assert ~torch.isnan(head_cr2).any()
        # assert ~torch.isnan(head_cc1).any()
        # assert ~torch.isnan(head_cc2).any()
        # assert ~torch.isnan(head_dc1).any()
        # assert ~torch.isnan(head_dc2).any()
        # assert ~torch.isnan(root_c).any()
            # assert ~torch.isnan(root_d).any()
            #
        return {'head_dd1': head_dd1,
                'head_cd1': head_cd1,
                'head_dd2': head_dd2,
                'head_cd2': head_cd2,
                'unary_l': unary_l,
                'unary_ld': unary_ld,
                'unary_r': unary_r,
                'unary_rd': unary_rd,
                'unary_c': unary_c,
                'unary_d': unary_d,

                'head_d1': head_d1,
                'head_d2': head_d2,

                'head_cl1': head_cl1,
                'head_cl2': head_cl2,
                'head_cr1': head_cr1,
                'head_cr2': head_cr2,

                'head_dl1': head_dl1,
                'head_dl2': head_dl2,
                'head_dr1': head_dr1,
                'head_dr2': head_dr2,

                'head_cc1': head_cc1,
                'head_cc2': head_cc2,
                'head_dc1': head_dc1,
                'head_dc2': head_dc2,
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



