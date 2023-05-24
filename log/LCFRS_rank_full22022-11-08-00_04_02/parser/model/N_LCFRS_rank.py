import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from ..pcfgs.lcfrs_simplest import PCFG

class NeuralLCFRS(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralLCFRS, self).__init__()
        self.pcfg = PCFG()
        self.device = dataset.device
        self.args = args

        self.NT = args.NT
        self.T = args.T
        self.D = args.D
        self.r = args.r
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim

        self.nonterm_emb = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))
        self.d_emb = nn.Parameter(torch.randn(self.D, self.s_dim))
        self.d_emb2 = nn.Parameter(torch.randn(self.D*4, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.rc_emb = nn.Parameter(torch.randn(self.s_dim, self.r))
        self.rd_emb = nn.Parameter(torch.randn(self.s_dim, self.r))

        self.ratio_c = int(0.95 * self.r)
        self.ratio_d = int(0.95 * self.r)

        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.root_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        self.left_c = nn.Sequential(nn.Linear(self.s_dim, self.s_dim))
        self.right_c = nn.Sequential(nn.Linear(self.s_dim, self.s_dim))
        self.parent_c = nn.Sequential(nn.Linear(self.s_dim, self.s_dim))
        self.cd_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim))
        self.cc_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim))

        self.parent_d = nn.Sequential(nn.Linear(self.s_dim, self.s_dim))
        self.left_d = nn.Sequential(nn.Linear(self.s_dim, self.s_dim))
        self.right_d = nn.Sequential(nn.Linear(self.s_dim, self.s_dim))
        self.dd_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim))
        self.dc_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim))

        self.NT_T = self.NT + self.T
        # self.NT_T_D = self.NT + self.T
        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.

        self._initialize()


    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

    def forward(self, input, evaluating=False):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            root_emb = self.root_emb
            roots = self.root_mlp(root_emb).log_softmax(-1)
            return roots.expand(b, self.NT)

        def terms():
            term_prob = self.term_mlp(self.nonterm_emb[self.NT:]).log_softmax(-1)
            return term_prob[torch.arange(self.T)[None,None], x[:, :, None]].contiguous()

        def rules():

            head = (self.parent_c(self.nonterm_emb[:self.NT]) @ self.rc_emb).softmax(-1)
            left_c = (self.left_c(self.nonterm_emb) @ self.rc_emb[:, :self.ratio_c]).softmax(0)
            right_c = (self.right_c(self.nonterm_emb) @ self.rc_emb[:, :self.ratio_c]).softmax(0)
            mid_c = (self.cc_mlp(self.nonterm_emb) @ self.rc_emb[:, self.ratio_c:]).softmax(0)
            mid_d = (self.cd_mlp(self.d_emb) @  self.rc_emb[:, self.ratio_c:]).softmax(0)

            binary = (torch.einsum('ar, br, cr->abc', head[:, :self.ratio_c], left_c, right_c) + 1e-9).log()
            binary_close = (torch.einsum('ar, br, cr->abc', head[:,self.ratio_c:],  mid_d, mid_c) + 1e-9).log()

            head_d = (self.parent_d(self.d_emb) @ self.rd_emb).softmax(-1)
            left_d = (self.left_d(self.nonterm_emb) @ self.rd_emb[:,:self.ratio_d]).softmax(0)
            right_d = (self.right_d(self.nonterm_emb) @ self.rd_emb[:, :self.ratio_d]).softmax(0)
            mid_c = (self.dc_mlp(self.nonterm_emb) @ self.rd_emb[:, self.ratio_d:]).softmax(0)
            mid_d = (self.dd_mlp(self.d_emb2).reshape(self.NT, 4, -1).reshape(self.NT*4, -1) @ self.rd_emb[:, self.ratio_d:]).softmax(0)
            binary_dc = (torch.einsum('ar, br, cr->abc', head_d[:,:self.ratio_d], left_d, right_d) + 1e-9).log()
            binary_d = (torch.einsum('ar, br, cr -> abc',head_d[:,self.ratio_d:], mid_d, mid_c) + 1e-9).log().reshape(self.D, self.D, 4, self.NT_T).transpose(-1, -2)

            binary = binary.unsqueeze(0).expand(b, self.NT, self.NT_T, self.NT_T).contiguous()
            binary_close = binary_close.unsqueeze(0).expand(b, self.NT, self.D, self.NT_T).contiguous()
            binary_dc = binary_dc.unsqueeze(0).expand(b, self.D, self.NT_T, self.NT_T).contiguous()
            binary_d = binary_d.unsqueeze(0).expand(b, self.D, self.D, self.NT_T, 4).contiguous()
            return binary, binary_close, binary_dc, binary_d

        (binary, binary_close, binary_dc, binary_d), unary, root = rules(), terms(), roots()

        return {'unary': unary,
                'root': root,
                'binary': binary,
                'binary_closed': binary_close,
                'binary_dc': binary_dc,
                'binary_d': binary_d,
                'kl': torch.tensor(0, device=self.device)}


    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg._inside(rules=rules, lens=input['seq_len'])
        return -result['partition'].mean()


    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

