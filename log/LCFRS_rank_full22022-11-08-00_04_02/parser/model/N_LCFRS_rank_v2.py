import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from ..pcfgs.lcfrs_on5_cpd import PCFG

class NeuralLCFRS_RankSpace(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralLCFRS_RankSpace, self).__init__()
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
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb2 = nn.Parameter(torch.randn((self.NT + self.T)*4, self.s_dim))
        self.d_emb = nn.Parameter(torch.randn(self.D, self.s_dim))
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

        self.left_c = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.right_c = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.parent_c = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.cd_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.cc_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())

        self.parent_d = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.left_d = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.right_d = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.dd_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.dc_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())

        self.NT_T = self.NT + self.T
        # self.NT_T_D = self.NT + self.T
        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.

        self._initialize()
        torch.nn.init.xavier_uniform_(self.term_emb)


    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

    def forward(self, input, evaluating=False):
        x = input['word']
        b, n = x.shape[:2]
        root_emb = self.root_emb
        roots = self.root_mlp(root_emb).log_softmax(-1).expand(b, self.NT)

        term_prob = self.term_mlp(self.nonterm_emb[self.NT:]).log_softmax(-1)
        unary = term_prob[torch.arange(self.T)[None,None], x[:, :, None]].contiguous()

        unary_max = unary.max(-1)[0]
        unary = (unary-unary_max.unsqueeze(-1)).exp()

        head = (self.parent_c(self.nonterm_emb[:self.NT]) @ self.rc_emb).softmax(-1)
        left_c = (self.left_c(self.nonterm_emb) @ self.rc_emb[:, :self.ratio_c]).softmax(0)
        right_c = (self.right_c(self.nonterm_emb) @ self.rc_emb[:, :self.ratio_c]).softmax(0)
        left_d = (self.left_d(self.nonterm_emb) @ self.rd_emb[:,:self.ratio_d]).softmax(0)
        right_d = (self.right_d(self.nonterm_emb) @ self.rd_emb[:, :self.ratio_d]).softmax(0)
        unary_lc =  ((unary @ left_c[self.NT:, :]) + 1e-9).log() + unary_max.unsqueeze(-1)
        unary_rc = ((unary @ right_c[self.NT:, :]) + 1e-9).log() + unary_max.unsqueeze(-1)
        unary_ld =  ((unary @ left_d[self.NT:, :]) + 1e-9).log() + unary_max.unsqueeze(-1)
        unary_rd = ((unary @ right_d[self.NT:, :]) + 1e-9).log() + unary_max.unsqueeze(-1)
        left_c = ((head.t() @ left_c[:self.NT, :]) + 1e-9).log()
        right_c = ((head.t() @ right_c[:self.NT, :]) + 1e-9).log()
        left_c1 = left_c[:self.ratio_c, :]
        left_c2 = left_c[self.ratio_c:, :]
        right_c1 = right_c[:self.ratio_c, :]
        right_c2 = right_c[self.ratio_c:, :]

        cc = (self.cc_mlp(self.nonterm_emb) @ self.rc_emb[:, self.ratio_c:]).softmax(0)
        dc = (self.dc_mlp(self.nonterm_emb2) @ self.rd_emb[:, self.ratio_d:]).softmax(0).reshape(self.NT_T, 4, -1)
        unary_cc = ((unary @ cc[self.NT:, :]) + 1e-9).log() + unary_max.unsqueeze(-1)
        cc = ((head.t() @ cc[:self.NT]) + 1e-9).log()
        cc_1 = cc[:self.ratio_c]
        cc_2 = cc[self.ratio_c:]
        unary_dc = ((unary_cc@ dc[self.NT:])+1e-9).log() + unary_max.unsqueeze(-1).unsqueeze(-1)
        dc = ((head.t() @ dc[:self.NT]) + 1e-9).log()
        dc_1 = dc[:self.ratio_c]
        dc_2 = dc[self.ratio_c:]

        head_d = (self.parent_d(self.d_emb) @ self.rd_emb).softmax(-1)
        dd = (self.dd_mlp(self.d_emb) @ self.rd_emb[:, self.ratio_d:]).softmax(0)
        cd = (self.cd_mlp(self.d_emb) @ self.rc_emb[:, self.ratio_c:]).softmax(0)
        dd = (head_d.t() @ dd + 1e-9).log()
        cd = (head_d.t() @ cd + 1e-9).log()

        dd_1 = dd[:self.ratio_d]
        dd_2 = dd[self.ratio_d:]
        cd_1 = cd[:self.ratio_d]
        cd_2 = cd[self.ratio_d:]

        return {'root': roots,
                'unary_lc': unary_lc,
                'unary_rc': unary_rc,
                'unary_ld': unary_ld,
                'unary_rd': unary_rd,

                'cc_1': cc_1,
                'cc_2': cc_2,
                'cd_1': cd_1,
                'cd_2': cd_2,

                'head_d1': head_d1,
                'head_d2': head_d2,
                'left_d': left_d,
                'right_d': right_d,
                'dc': dc,
                'dd': dd,
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

