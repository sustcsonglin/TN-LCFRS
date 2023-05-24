import pdb
import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from parser.lcfrs_cuda.lcfrs_on5_cpd import LCFRS

from parser.lcfrs_triton.merge_continuous import _merge_continuous
from parser.lcfrs_triton.merge_discontinuous import _merge_discontinuous
from parser.lcfrs_triton.save_continuous import _save_continous
from parser.lcfrs_triton.save_discontinuous import _save_discontinuous
from parser.mbr_decoding import mbr_decoding



class TNLCFRS(nn.Module):
    def __init__(self, args, dataset):
        super(TNLCFRS, self).__init__()
        self.lcfrs = LCFRS()
        self.device = dataset.device
        self.args = args
        # self.activate =
        self.NT = args.NT
        self.T = args.T
        self.activate = nn.ReLU()

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
        tmp = self.s_dim
        self.decision = nn.Sequential(
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, 4))
        self.left_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.left_d = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.right_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.right_d = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.parent_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.parent_d = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)

        self.cd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.cc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.r1_emb = nn.Parameter(torch.randn(self.s_dim, self.r1))
        self.r2_emb = nn.Parameter(torch.randn(self.s_dim, self.r2))
        self.r3_emb = nn.Parameter(torch.randn(self.s_dim, self.r3))
        self.r4_emb = nn.Parameter(torch.randn(self.s_dim, self.r4))

        self.NT_T = self.NT + self.T
        self.vocab_emb = nn.Parameter(torch.randn(self.V, self.s_dim))
        self._initialize()

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
                                                                                                                      1,                                                                                                              1).contiguous()
        head_cl2 = (torch.einsum('xi, xj -> ji', head_c2, left_c[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b,1,                            1,
                                                                                                                      1).contiguous()

        head_dl1 = (torch.einsum('xi, xj -> ji', head_c1, left_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()
        head_dr1 = (torch.einsum('xi, xj -> ji', head_c1, right_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()

        head_dl2 = (torch.einsum('xi, xj -> ji', head_c2, left_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()
        head_dr2 = (torch.einsum('xi, xj -> ji', head_c2, right_d[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,1).contiguous()

        head_cc1 = (torch.einsum('xi, xj -> ji', head_c1, cc[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,
                                                                                                            1).contiguous()
        head_cc2 = (torch.einsum('xi, xj -> ji', head_c2, cc[..., :self.NT, :]) + 1e-9).log().unsqueeze(0).repeat(b, 1,                                                                                                                  1).contiguous()

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
        result = self.lcfrs._inside(rules=rules, lens=input['seq_len'])
        return result['partition']

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            return self.lcfrs.decode(rules=rules, lens=input['seq_len'], raw_word=input['raw_word'], viterbi=True, mbr=False)
        elif decode_type == 'mbr':
            return self.lcfrs.decode(rules=rules, lens=input['seq_len'],  raw_word=input['raw_word'],viterbi=False, mbr=True)
        else:
            raise NotImplementedError





#Triton's version, experimental
class TNLCFRS_Triton(nn.Module):
    def __init__(self, args, dataset):
        super(TNLCFRS_Triton, self).__init__()
        self.lcfrs = LCFRS()


        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.activate = nn.ReLU()
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

        tmp = self.s_dim

        self.decision = nn.Sequential(
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, 4))

        self.left_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.left_d = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.right_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.right_d = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.parent_c = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.parent_d = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)

        self.cd_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)
        self.cc_mlp = nn.Sequential(nn.Linear(self.s_dim, tmp),self.activate)

        self.r1_emb = nn.Parameter(torch.randn(self.s_dim, self.r1))
        self.r2_emb = nn.Parameter(torch.randn(self.s_dim, self.r2))
        self.r3_emb = nn.Parameter(torch.randn(self.s_dim, self.r3))
        self.r4_emb = nn.Parameter(torch.randn(self.s_dim, self.r4))

        self.NT_T = self.NT + self.T
        self.vocab_emb = nn.Parameter(torch.randn(self.V, self.s_dim))
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @torch.enable_grad()
    def forward(self, input, evaluating=False):
        x = input['word']
        B, n = x.shape[:2]
        r1_emb = self.r1_emb
        r2_emb = self.r2_emb
        r3_emb = self.r3_emb
        r4_emb = self.r4_emb
        d_emb = self.d_emb
        # root
        root_emb = self.root_emb
        root = self.root_mlp(root_emb).softmax(-1).expand(B, -1)
        # unary
        term_prob = (self.term_mlp(self.nonterm_emb[self.NT:])  @ self.vocab_emb.t()).log_softmax(-1)
        unary = term_prob[torch.arange(self.T)[None, None], x[:, :, None]]
        # if evaluating:
            # unary = unary.clone().requires_grad_(True)
        head_c = self.parent_c(self.nonterm_emb[:self.NT])
        head_d = self.parent_d(d_emb)
        left = self.left_c(self.nonterm_emb)
        right = self.right_c(self.nonterm_emb)
        C = self.cc_mlp(self.nonterm_emb)
        D = self.cd_mlp(d_emb)
        # (nt, r1+r2)
        head = torch.cat([head_c @ self.r1_emb, head_c @ self.r2_emb], -1).softmax(-1)
        # (b, nt) @ (nt, r1+r2) = (b, r1+r2)
        root = root @ head
        # (nt, r3+r4)
        head_d = torch.cat([head_d @ self.r3_emb, head_d @ self.r4_emb], -1).softmax(-1)

        # （nt+t, r1)
        left_c = (left @ r1_emb).softmax(0)
        # (nt+t, r1)
        right_c = (right @ r1_emb).softmax(0)

        # (nt+t, r2)
        cc = (C @ r2_emb).softmax(0)

        # (nt+t, r3)
        left_d = (left @ r3_emb).softmax(0)
        # (nt+t, r3)
        right_d = (right @ r3_emb).softmax(0)

        # (nt+t, 4 * r4) 
        dc = (C @ r4_emb).softmax(0)
        decision = self.decision(r4_emb.t()).softmax(-1)
        dc = torch.einsum('cr, rd -> cdr', dc, decision).flatten(start_dim=-2)
        
        # (nt+t, 2*r1+r2+2*r3+4*r4)
        all = torch.cat([left_c, right_c, cc, left_d, right_d, dc], dim=-1)
        
        #  (r1+r2, nt) @  (nt, 2*r1+r2+2*r3+4*r4) = (r1+r2,  2*r1+r2+2*r3+4*r4)
        f_c = head.t() @ all[:self.NT, :]
        # (t, 2*r1+r2+2*r3+4*r4)
        f_p = all[self.NT:, :]                                
        # (r3+r4, nt) @ (nt, r2+r4) = (r3+r4, r2+r4)
        f_d = head_d.t() @ torch.cat([(D @ r2_emb).softmax(0), 
                         (D @ r4_emb).softmax(0)
                         ], dim=-1)        
        f_d1 = f_d[:self.r3]
        f_d2 = f_d[self.r3:]
        

        r_c = self.r1 * 2 + self.r2 + self.r3 * 2 + self.r4 * 4
        r_d = self.r2 + self.r4
        N = n + 1
    
        with torch.no_grad():
            normalizer = unary.max(-1)[0]

        out = (unary - normalizer.unsqueeze(-1)).exp()
        out = torch.einsum('blr, rq -> blq', out, f_p)                

        alpha_c = unary.new_zeros(B, N, N, r_c)
        alpha_d = unary.new_zeros(B, N, N, N, N, r_d)        
        
        alpha_c = _save_continous(out, normalizer, alpha_c)

        for w in range(2, N):
            n = N - w      
            # tmp = alpha_c.new_zeros(B, n, w-1, r)
            # out = alpha_c.new_zeros(B, n, r)
            dimension_info = torch.tensor([ w, self.r1, self.r2, self.r3, self.r4], device=alpha_c.device)
            out, normalizer = _merge_continuous(alpha_c, alpha_d, dimension_info)
            if w < N-1:                                
                out = torch.einsum('blr, rq -> blq', out, f_c)                
                alpha_c = _save_continous(out, normalizer, alpha_c)
                out = _merge_discontinuous(alpha_c, alpha_d, f_d1, f_d2, dimension_info)
                alpha_d = _save_discontinuous(out, alpha_d, dimension_info)
                
        logZ = (torch.einsum('bnr, br -> b', out, root) + 1e-9).log() + normalizer.squeeze(1)        
        
        if not evaluating:
            return logZ
        else:
            logZ.sum().backward()
            return logZ, alpha_c, alpha_d
            
    def loss(self, input):
        logZ = self.forward(input)
        # result = self.pcfg._inside(rules=rules, lens=input['seq_len'])        
        # pdb.set_trace()
        return logZ


    def evaluate(self, input, decode_type, **kwargs):
        logZ, alpha_c, alpha_d = self.forward(input, evaluating=True)
        lens=input['seq_len']
        raw_word=input['raw_word']    
        marginal_c = alpha_c.sum(-1).transpose(1,2).contiguous()
        marginal_d = alpha_d.sum(-1).transpose(1,2).contiguous()
        prediction, predicted_trees = mbr_decoding(marginal_c, marginal_d, raw_word, lens)
    
    
        return {'prediction': prediction,
                'partition': logZ,
                'prediction_tree': predicted_trees}

        

