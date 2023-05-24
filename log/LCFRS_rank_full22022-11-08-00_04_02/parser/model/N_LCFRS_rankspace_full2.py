import pdb

import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from ..pcfgs.lcfrs_on5_cpd_rankspace_full2 import PCFG

def Nothing(x):
    return x


class NeuralLCFRS_rankspace_full2(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralLCFRS_rankspace_full2, self).__init__()
        self.pcfg = PCFG()
        self.device = dataset.device
        self.args = args
        # self.activate =
        self.NT = args.NT
        self.T = args.T
        self.activate = nn.ReLU()
    # else:
    #         raise NotImplementedError

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

        # assert self.r4 <= self.r2

        self.use_char = self.args.use_char
        self.share_r = self.args.share_r
        self.sep_emb = self.args.sep_emb

        self.term_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
        )


        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT)
        )

        # self.d_project =
        tmp = self.s_dim

        self.decision = nn.Sequential(
                                    # nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                    # nn.ReLU(),
                                      nn.Linear(self.s_dim, 4))

        self.left_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.left_d = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.right_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.right_d = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.parent_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.parent_d = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)

        # self.parent_c2 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.cd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.cc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        # self.parent_d2 = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
        # self.left_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
        # self.right_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
        # self.dd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        # self.dc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        # self.dc_mlp1 = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate)
        # self.dc_mlp2 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        # self.dc_mlp3 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        # self.dc_mlp4 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.r1_emb = nn.Parameter(torch.randn(self.s_dim, self.r1))
        self.r2_emb = nn.Parameter(torch.randn(self.s_dim, self.r2))
        self.r3_emb = nn.Parameter(torch.randn(self.s_dim, self.r3))
        self.r4_emb = nn.Parameter(torch.randn(self.s_dim, self.r4))
            #
            # # self.right_d = nn.Sequential(nn.Linear(self.s_dim, tmp), ResLayer(self.s_dim, tmp), self.activate)
            # # self.dd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), ResLayer(self.s_dim, tmp),self.activate)
            # # self.dc_mlp = nn.Sequential(ResLayer(self.s_dim, tmp),self.activate)
            # # self.dc_mlp1 = nn.Sequential(ResLayer(self.s_dim, tmp), self.activate)
            # # self.dc_mlp2 = nn.Sequential(ResLayer(self.s_dim, tmp),self.activate)
            # # self.dc_mlp3 = nn.Sequential(ResLayer(self.s_dim, tmp),self.activate)
            # # self.dc_mlp4 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
            # self.r1_emb = nn.Parameter(torch.randn(self.s_dim, self.r1))
            # self.r2_emb = nn.Parameter(torch.randn(self.s_dim, self.r2))
            # self.r3_emb = nn.Parameter(torch.randn(self.s_dim, self.r3))
            # self.r4_emb = nn.Parameter(torch.randn(self.s_dim, self.r4))

        # self.nonterm_emb2 = nn.Parameter(torch.randn((self.NT + self.T) * 4, self.s_dim))

        self.NT_T = self.NT + self.T
        # self.NT_T_D = self.NT + self.T
        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        #
        # if dataset.emb is None:
        self.vocab_emb = nn.Parameter(torch.randn(self.V, self.s_dim))
        # torch.nn.init.xavier_uniform_(self.vocab_emb)
        self._initialize()

        # else:
        #
        #     self.vocab_emb = nn.Parameter(torch.from_numpy(dataset.emb).float())

        # torch.nn.init.normal_(self.term_emb)

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


    def forward(self, input, evaluating=False):
        x = input['word']
        b, n = x.shape[:2]

        r1_emb = self.r1_emb
        r2_emb = self.r2_emb
        r3_emb = self.r3_emb
        r4_emb = self.r4_emb
        d_emb = self.d_emb

        root_emb = self.root_emb
        root = self.root_mlp(root_emb).softmax(-1).expand(b, -1)

        # if not self.args.share_pt:
        #     term_prob = (self.term_mlp(self.term_emb) @ self.vocab_emb.t()).softmax(-1)
        # else:
        term_prob = (self.term_mlp(self.nonterm_emb[self.NT:])  @ self.vocab_emb.t()).softmax(-1)
        unary = term_prob[torch.arange(self.T)[None, None], x[:, :, None]]

        head_c = self.parent_c(self.nonterm_emb[:self.NT])
        head_d = self.parent_d(d_emb)

        left = self.left_c(self.nonterm_emb)
        right = self.right_c(self.nonterm_emb)
        C = self.cc_mlp(self.nonterm_emb)
        D = self.cd_mlp(d_emb)
        head = torch.cat([head_c @ self.r1_emb, head_c @ self.r2_emb], -1).softmax(-1)
        head_c1 = head[:, :self.r1]
        head_c2 = head[:, self.r1:]
        head_d = torch.cat([head_d @ self.r3_emb, head_d @ self.r4_emb], -1).softmax(-1)
        head_d1 = head_d[:, :self.r3]
        head_d2 = head_d[:, self.r3:]
        left_c = (left @ r1_emb).softmax(0)
        right_c = (right @ r1_emb).softmax(0)
        left_d = (left @ r3_emb).softmax(0)
        right_d = (right @ r3_emb).softmax(0)
        cc = (C @ r2_emb).softmax(0)
        dc = (C @ r4_emb).softmax(0)
        cd = (D @ r2_emb).softmax(0)
        dd = (D @ r4_emb).softmax(0)
        decision = self.decision(r4_emb.t()).softmax(-1)
        dc = torch.einsum('cr, rd -> cdr', dc, decision)

        # dd = cd
        head_cd1 = (torch.einsum('xi, xj -> ji', head_d1, cd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()
        head_cd2 = (torch.einsum('xi, xj -> ji', head_d2, cd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()
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


    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], raw_word=input['raw_word'], viterbi=True, mbr=False)
        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'],  raw_word=input['raw_word'],viterbi=False, mbr=True)
        else:
            raise NotImplementedError





