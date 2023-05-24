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
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim

        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.d_emb = nn.Parameter(torch.randn(self.D, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.root_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        self.NT_T = self.NT + self.T
        # self.NT_T_D = self.NT + self.T
        self.rule_mlp = nn.Linear(self.s_dim, (self.NT_T) * self.NT_T + self.NT_T * self.D)
        self.rule_mlp2 = nn.Linear(self.s_dim, self.D * self.NT_T * 4 + self.NT_T*self.NT_T)
        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        self._initialize()


    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input, evaluating=False):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            root_emb = self.root_emb
            roots = self.root_mlp(root_emb).log_softmax(-1)
            return roots.expand(b, self.NT)

        def terms():
            term_prob = self.term_mlp(self.term_emb).log_softmax(-1)
            return term_prob[torch.arange(self.T)[None,None], x[:, :, None]]

        def rules():
            rule_prob = self.rule_mlp(self.nonterm_emb).log_softmax(-1)
            binary = rule_prob[:, :self.NT_T*self.NT_T].reshape(1,self.NT, self.NT_T, self.NT_T).expand(b, self.NT, self.NT_T, self.NT_T).contiguous()
            binary_close = rule_prob[:, self.NT_T*self.NT_T:].reshape(1,self.NT, self.D, self.NT_T).expand(b, self.NT, self.D, self.NT_T).contiguous()

            rule_prob2 = self.rule_mlp2(self.d_emb).log_softmax(-1)
            binary_dc = rule_prob2[:, :self.NT_T*self.NT_T].reshape(1,self.D, self.NT_T, self.NT_T).expand(b, self.D, self.NT_T, self.NT_T).contiguous()
            binary_d = rule_prob2[:, self.NT_T*self.NT_T:].reshape(1,self.D, self.D, self.NT_T, 4).expand(b, self.D, self.D, self.NT_T, 4).contiguous()
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

