import pdb

import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from parser.modules.charRNN import CharProbRNN
from ..pcfgs.lcfrs_on5_cpd_rankspace_full import PCFG



class NeuralLCFRS_rankspace_full(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralLCFRS_rankspace_full, self).__init__()
        self.pcfg = PCFG()
        self.device = dataset.device
        self.args = args
        self.activate = nn.Tanh()
        # else:
        #     raise NotImplementedError

        self.NT = args.NT
        self.T = args.T
        self.D = args.D
        self.r = args.r1 + args.r2
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim

        self.nonterm_emb = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))
        self.nonterm_emb_parent = nn.Parameter(torch.randn(self.NT , self.s_dim))
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.d_emb = nn.Parameter(torch.randn(self.D, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.ratio_c =  self.args.r1
        self.ratio_d = self.ratio_c

        self.use_char = self.args.use_char
        self.share_r = self.args.share_r
        self.sep_emb = self.args.sep_emb

        self.term_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
        )

        self.split_prob = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, 4)
        )

        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT))

        tmp = self.s_dim


        if not self.args.share_r:

            self.left_c = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate, nn.Linear(tmp, self.ratio_c))
            self.right_c = nn.Sequential(nn.Linear(self.s_dim, tmp),  self.activate, nn.Linear(tmp, self.ratio_c))
            self.parent_c = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate,   nn.Linear(tmp, self.r))
            self.cd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate,nn.Linear(tmp, self.r - self.ratio_c))
            self.cc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),  self.activate, nn.Linear(tmp, self.r - self.ratio_c))

            self.parent_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate, nn.Linear(tmp, self.r))
            self.left_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate,  nn.Linear(tmp, self.ratio_c))
            self.right_d = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate, nn.Linear(tmp, self.ratio_c))
            self.dd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate,  nn.Linear(tmp, self.r - self.ratio_c))

            self.dc_mlp = nn.Sequential(nn.Linear(self.s_dim, (self.r - self.ratio_c)))
            self.dc_mlp1 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate,  nn.Linear(tmp, (self.r - self.ratio_c)))
            self.dc_mlp2 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate,  nn.Linear(tmp, (self.r - self.ratio_c)))
            self.dc_mlp3 = nn.Sequential(nn.Linear(self.s_dim, tmp), self.activate, nn.Linear(tmp, (self.r - self.ratio_c)))
            self.dc_mlp4 = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate,  nn.Linear(tmp, (self.r - self.ratio_c)))

        else:
            self.left_c = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())
            self.right_c = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())
            self.parent_c = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())
            self.cd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())
            self.cc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())

            self.parent_d = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())
            self.left_d = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())
            self.right_d = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())
            self.dd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())
            self.dc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())

            self.dc_mlp1 = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())
            self.dc_mlp2 = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())
            self.dc_mlp3 = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())
            self.dc_mlp4 = nn.Sequential(nn.Linear(self.s_dim, tmp), nn.ReLU())

            self.r_emb = nn.Parameter(torch.randn(self.s_dim, self.r))

        self.nonterm_emb2 = nn.Parameter(torch.randn((self.NT + self.T) * 4, self.s_dim))

        self.NT_T = self.NT + self.T
        # self.NT_T_D = self.NT + self.T
        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        #
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
            r1_emb = self.r_emb[:, :self.ratio_c]
            r2_emb = self.r_emb[:, self.ratio_c:]

        d_emb = self.d_emb

        root_emb = self.root_emb
        root = self.root_mlp(root_emb).softmax(-1).expand(b, -1)

        if not self.args.share_pt:
            term_prob = (self.term_mlp(self.term_emb) @ self.vocab_emb.t()).softmax(-1)
        else:
            term_prob = (self.term_mlp(self.nonterm_emb[self.NT:])  @ self.vocab_emb.t()).softmax(-1)

        unary = term_prob[torch.arange(self.T)[None, None], x[:, :, None]]

        # if not self.share_r:
        head = (self.parent_c(self.nonterm_emb[:self.NT])).softmax(-1)
        # else:
        #     head = (self.parent_c(self.nonterm_emb[:self.NT]) @ self.r_emb).softmax(-1)
        head_c1 = head[:, :self.ratio_c]
        head_c2 = head[:, self.ratio_c:]

        if not self.share_r:
            left_c = (self.left_c(self.nonterm_emb)).softmax(0)
            right_c = (self.right_c(self.nonterm_emb)).softmax(0)
            cc = (self.cc_mlp(self.nonterm_emb)).softmax(0)
            cd = (self.cd_mlp(d_emb)).softmax(0)

        else:
            left_c = (self.left_c(self.nonterm_emb) @ r1_emb).softmax(0)
            right_c = (self.right_c(self.nonterm_emb) @ r1_emb).softmax(0)
            cc = (self.cc_mlp(self.nonterm_emb) @ r2_emb).softmax(0)
            cd = (self.cd_mlp(d_emb) @ r2_emb).softmax(0)

        if not self.share_r:
            head_d = self.parent_d(d_emb)

            if self.args.ban_nested:
                mask = torch.zeros_like(head_d)
                mask[:, self.ratio_c:] = -1e9
                head_d = head_d + mask

            head_d = head_d.softmax(-1)

            if not self.sep_emb:
                dc1 = self.dc_mlp1(self.nonterm_emb)
                dc2 = self.dc_mlp2(self.nonterm_emb)
                dc3 = self.dc_mlp3(self.nonterm_emb)
                dc4 = self.dc_mlp4(self.nonterm_emb)
                dc = torch.cat([dc1, dc2, dc3, dc4], dim=0).softmax(0).reshape(self.NT_T, 4, -1)

            else:
                dc = self.dc_mlp(self.nonterm_emb2).softmax(0).reshape(self.NT_T, 4, -1)
            dd = self.dd_mlp(self.d_emb).softmax(0)

        else:
            head_d = (self.parent_d(d_emb) @ self.r_emb).softmax(-1)
            dc = (self.dc_mlp(self.nonterm_emb2) @ r2_emb).softmax(0).reshape(self.NT_T, 4, -1)
            dd = (self.dd_mlp(self.d_emb) @ r2_emb).softmax(0)


        head_d1 = head_d[:, :self.ratio_d].contiguous()
        head_d2 = head_d[:, self.ratio_d:].contiguous()

        # dd = cd

        head_cd1 = (torch.einsum('xi, xj -> ij', head_d1, cd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()
        head_cd2 = (torch.einsum('xi, xj -> ij', head_d2, cd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()
        head_dd1 = (torch.einsum('xi, xj -> ij', head_d1, dd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()
        head_dd2 = (torch.einsum('xi, xj -> ij', head_d2, dd) + 1e-9).log().unsqueeze(0).repeat(b, 1, 1).contiguous()

        unary_l = (torch.einsum('blp, pr -> blr', unary, left_c[..., self.NT:, :]) + 1e-9).log()
        unary_r = (torch.einsum('blp, pr -> blr', unary, right_c[..., self.NT:, :]) + 1e-9).log()
        unary_c = (torch.einsum('blp, pr -> blr', unary, cc[..., self.NT:, :]) + 1e-9).log()
        unary_d = (torch.einsum('blp, pdr -> bldr', unary, dc[self.NT:, :, :]) + 1e-9).log()

        head_cl1 = (torch.einsum('xi, xj -> ji', head_c1, left_c[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b,
                                                                                                                      1,
                                                                                                                      1).contiguous()
        head_cl2 = (torch.einsum('xi, xj -> ji', head_c2, left_c[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b,
                                                                                                                      1,
                                                                                                                      1).contiguous()

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
        assert ~torch.isnan(head_dd1).any()
        assert ~torch.isnan(head_cd1).any()
        assert ~torch.isnan(head_dd2).any()
        assert ~torch.isnan(head_cd2).any()
        assert ~torch.isnan(unary_l).any()
        assert ~torch.isnan(unary_r).any()
        assert ~torch.isnan(unary_c).any()
        assert ~torch.isnan(unary_d).any()
        assert ~torch.isnan(head_cl1).any()
        assert ~torch.isnan(head_cl2).any()
        assert ~torch.isnan(head_cr1).any()
        assert ~torch.isnan(head_cr2).any()
        assert ~torch.isnan(head_cc1).any()
        assert ~torch.isnan(head_cc2).any()
        assert ~torch.isnan(head_dc1).any()
        assert ~torch.isnan(head_dc2).any()
        assert ~torch.isnan(root_c).any()
        assert ~torch.isnan(root_d).any()

        return {'head_dd1': head_dd1,
                'head_cd1': head_cd1,
                'head_dd2': head_dd2,
                'head_cd2': head_cd2,
                'unary_l': unary_l,
                'unary_r': unary_r,
                'unary_c': unary_c,
                'unary_d': unary_d,

                'head_cl1': head_cl1,

                'head_cl2': head_cl2,
                'head_cr1': head_cr1,
                'head_cr2': head_cr2,
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
        raise NotImplementedError
        rules = self.forward(input)
        rules2 = model2.forward(input)
        with torch.no_grad():
            head_c1_grd, head_c2_grd, left_c_grd, right_c_grd, left_d_grd, right_d_grd, \
            cc_grd, dc_grd, head_cd1_grd, head_cd2_grd, head_dd1_grd, head_dd2_grd, unary_grd, root_grd = self.pcfg.compute_marginals(
                rules2, input['seq_len'], False)
        loss = (rules['unary'] * unary_grd).sum() + (rules['head_c1'] * head_c1_grd).sum() \
               + (rules['head_c2'] * head_c2_grd).sum() + (rules['left_c'] * left_c_grd).sum() \
               + (rules['right_c'] * right_c_grd).sum() + (rules['left_d'] * left_d_grd).sum() \
               + (rules['right_d'] * right_d_grd).sum() + (rules['cc'] * cc_grd).sum() + (rules['dc'] * dc_grd).sum() \
               + (rules['head_cd1'] * head_cd1_grd).sum() + (rules['head_cd2'] * head_cd2_grd).sum() \
               + (rules['head_dd1'] * head_dd1_grd).sum() + (rules['head_dd2'] * head_dd2_grd).sum() \
               + (rules['root'] * root_grd).sum()
        loss = -loss / head_c1_grd.shape[0]
        return loss

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

