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

        self.nonterm_emb = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))
        self.d_emb = nn.Parameter(torch.randn(self.D, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.type_emb = nn.Parameter(torch.randn(5, self.s_dim))

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


        self.rule_parent = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU(), nn.Linear(self.s_dim, 100), nn.Linear(100, 100))
        self.rule_left = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU(), nn.Linear(self.s_dim, 100), nn.Linear(100, 100))
        self.rule_mid_c = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU(), nn.Linear(self.s_dim, 100), nn.Linear(100, 100))
        self.rule_mid_d = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU(), nn.Linear(self.s_dim, 100), nn.Linear(100, 100))
        self.rule_mid_t = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU(), nn.Linear(self.s_dim, 100), nn.Linear(100, 100))
        self.rule_right = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU(), nn.Linear(self.s_dim, 100), nn.Linear(100, 100))


        # I find this is important for neural/compound PCFG. if do not use this initialization, the performance would get much worser.
        # self._initialize()

    def forward(self, input, evaluating=False):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            root_emb = self.root_emb
            roots = self.root_mlp(root_emb).log_softmax(-1)
            return roots.expand(b, self.NT)

        def terms():
            term_prob = self.term_mlp(self.nonterm_emb[self.NT:]).log_softmax(-1)
            return term_prob[torch.arange(self.T)[None,None], x[:, :, None]]

        def rules():
            type = self.rule_t(self.type_emb)

            parent_nt = self.rule_parent(self.nonterm_emb[:self.NT])
            left_c = self.rule_left(self.nonterm_emb)
            right_c = self.rule_right(self.nonterm_emb)

            mid_c = self.rule_mid_c(self.nonterm_emb)
            mid_d = self.rule_mid_d(self.d_emb)
            abc = torch.einsum('ar, br, cr -> abc', parent_nt, left_c, right_c)
            adc = torch.einsum('ar, br, cr, r -> abc', parent_nt, mid_d, mid_c, type[0])
            rule_prob = torch.cat([abc.reshape(self.NT,-1), adc.reshape(self.NT, -1)], dim=1).log_softmax(-1)
            binary = rule_prob[:, :self.NT_T*self.NT_T].reshape(1,self.NT, self.NT_T, self.NT_T).expand(b, self.NT, self.NT_T, self.NT_T).contiguous()
            binary_close = rule_prob[:, self.NT_T*self.NT_T:].reshape(1,self.NT, self.D, self.NT_T).expand(b, self.NT, self.D, self.NT_T).contiguous()

            parent_d = self.rule_parent(self.d_emb)

            dbc = torch.einsum('ar, br, cr -> abc', parent_d, left_c, right_c).reshape(self.D, -1)
            dbc2 = torch.einsum('ar, br, cr, dr -> abcd', parent_d, mid_d, mid_c, type[1:]).reshape(self.D, -1)
            rule_prob2= torch.cat([dbc, dbc2],-1).log_softmax(-1)
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

