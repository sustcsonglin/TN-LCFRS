import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from ..pcfgs.lcfrs_on5_nopt import PCFG
from parser.modules.charRNN import CharProbRNN


class NeuralLCFRS_nopt(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralLCFRS_nopt, self).__init__()
        self.pcfg = PCFG()
        self.device = dataset.device
        self.args = args
        self.use_char = self.args.use_char
        self.NT = args.NT
        # self.D = args.D
        # self.r = args.r
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim

        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        if not self.use_char:
            self.term_mlp = nn.Sequential(
                nn.Linear(self.s_dim, self.s_dim),
                ResLayer(self.s_dim, self.s_dim),
                ResLayer(self.s_dim, self.s_dim),
                nn.Linear(self.s_dim, self.V))

        else:
            self.term_mlp = CharProbRNN(dataset.char_vocab_size, state_dim=self.s_dim)

        self.root_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, self.NT))

        self.split_mlp = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            ResLayer(self.s_dim, self.s_dim),
            nn.Linear(self.s_dim, 2))

        # self.split_mlp2 = nn.Sequentia    l(
        #     nn.Linear(self.s_dim, self.s_dim),
        #     ResLayer(self.s_dim, self.s_dim),
        #     ResLayer(self.s_dim, self.s_dim),
        #     nn.Linear(self.s_dim, 5))

        # self.binary = nn.Linear(self.s_dim, (self.NT ** 2) * 7)
        self.rule_mlp = nn.Linear(self.s_dim, (self.NT) * self.NT * 2)
        self.rule_mlp2 = nn.Linear(self.s_dim, self.NT * self.NT * 5)

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

        def rules():
            if not self.use_char:
                term_prob = self.term_mlp(self.nonterm_emb).log_softmax(-1)
                term_prob = term_prob[torch.arange(self.NT)[None, None], x[:, :, None]].contiguous()
            else:
                term_prob = self.term_mlp(input['char'], self.nonterm_emb)

            split_prob = self.split_mlp(self.nonterm_emb).log_softmax(-1)
            # split_prob2 = self.split_mlp2(self.nonterm_emb).log_softmax(-1)
            term_prob = (term_prob + split_prob[None, None, :, 0]).contiguous()

            binary_c = (self.rule_mlp(self.nonterm_emb).log_softmax(-1) + split_prob[:, 1:2])

            binary = (binary_c[:, :self.NT**2]).reshape(self.NT, self.NT, self.NT)
            binary_close = (binary_c[:, self.NT**2:]).reshape(self.NT, self.NT, self.NT)

            binady_d = self.rule_mlp2(self.nonterm_emb).log_softmax(-1)

            binary_dc = (binady_d[:, :self.NT**2]).reshape(self.NT, self.NT, self.NT)
            binary_d = (binady_d[:, self.NT**2:] ).reshape(self.NT,  self.NT, self.NT, 4)

            binary = binary.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
            binary_close = binary_close.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
            binary_dc = binary_dc.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
            binary_d = binary_d.unsqueeze(0).expand(b, -1, -1, -1, -1).contiguous()
            return term_prob, binary, binary_close, binary_dc, binary_d

        (unary, binary, binary_close, binary_dc, binary_d), root = rules(), roots()

        return {'unary': unary,
                'root': root,
                'binary': binary,
                'binary_closed': binary_close,
                'binary_dc': binary_dc,
                'binary_d': binary_d,
                'kl': torch.tensor(0, device=self.device)}

    def loss(self, input):
        rules = self.forward(input)
        result = self.pcfg._inside(rules=rules, lens=input['seq_len'])
        return -result['partition'].mean()

    def loss_em(self, input, model2):
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

