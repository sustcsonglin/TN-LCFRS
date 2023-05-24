
import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from ..pcfgs.lcfrs_simplest import PCFG


class CompoundLCFRS(nn.Module):
    def __init__(self, args, dataset):
        super(CompoundLCFRS, self).__init__()

        self.pcfg = PCFG()
        self.device = dataset.device
        self.args = args
        self.NT = args.NT
        self.T = args.T
        self.D = args.D
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim
        self.z_dim = args.z_dim
        self.w_dim = args.w_dim
        self.h_dim = args.h_dim

        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.d_emb = nn.Parameter(torch.randn(self.D, self.s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT , self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        input_dim = self.s_dim + self.z_dim

        self.term_mlp = nn.Sequential(nn.Linear(input_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.root_mlp = nn.Sequential(nn.Linear(input_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        self.enc_emb = nn.Embedding(self.V, self.w_dim)

        self.enc_rnn = nn.LSTM(self.w_dim, self.h_dim, bidirectional=True, num_layers=1, batch_first=True)

        self.enc_out = nn.Linear(self.h_dim * 2, self.z_dim * 2)

        self.NT_T = self.NT + self.T
        self.rule_mlp = nn.Linear(input_dim, (self.NT_T) * self.NT_T + self.NT_T * self.D)
        self.rule_mlp2 = nn.Linear(input_dim, self.D * self.NT_T * 4 + self.NT_T*self.NT_T)
        self._initialize()


    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input, evaluating=False):
        x = input['word']
        b, n = x.shape[:2]
        seq_len = input['seq_len']

        def enc(x):
            x_embbed = self.enc_emb(x)
            x_packed = pack_padded_sequence(
                x_embbed, seq_len.cpu(), batch_first=True, enforce_sorted=False
            )
            h_packed, _ = self.enc_rnn(x_packed)
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
            out = self.enc_out(h)
            mean = out[:, : self.z_dim]
            lvar = out[:, self.z_dim :]
            return mean, lvar

        def kl(mean, logvar):
            result = -0.5 * (logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)
            return result

        mean, lvar = enc(x)
        z = mean

        if not evaluating:
            z = mean.new(b, mean.size(1)).normal_(0,1)
            z = (0.5 * lvar).exp() * z + mean


        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            root_emb = torch.cat([root_emb, z], -1)
            roots = self.root_mlp(root_emb).log_softmax(-1)
            return roots

        def terms():
            term_emb = self.term_emb.unsqueeze(0).expand(
                b, self.T, self.s_dim
            )
            z_expand = z.unsqueeze(1).expand(b, self.T, self.z_dim)
            term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            return term_prob.gather(-1, x.unsqueeze(1).expand(b, self.T, x.shape[-1])).transpose(-1, -2).contiguous()

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim
            )
            z_expand = z.unsqueeze(1).expand(
                b, self.NT, self.z_dim
            )
            nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)

            d_emb = self.d_emb.unsqueeze(0).expand(
                b, self.D, self.s_dim
            )
            z_expand = z.unsqueeze(1).expand(
                b, self.D, self.z_dim
            )
            d_emb = torch.cat([d_emb, z_expand], -1)

            rule_prob = self.rule_mlp(nonterm_emb).log_softmax(-1)
            binary = rule_prob[:, :, :self.NT_T*self.NT_T].reshape(b,self.NT, self.NT_T, self.NT_T).expand(b, self.NT, self.NT_T, self.NT_T).contiguous()
            binary_close = rule_prob[:, :, self.NT_T*self.NT_T:].reshape(b,self.NT, self.D, self.NT_T).expand(b, self.NT, self.D, self.NT_T).contiguous()

            rule_prob2 = self.rule_mlp2(d_emb).log_softmax(-1)
            binary_dc = rule_prob2[:, :, :self.NT_T*self.NT_T].reshape(b,self.D, self.NT_T, self.NT_T).expand(b, self.D, self.NT_T, self.NT_T).contiguous()
            binary_d = rule_prob2[:, :, self.NT_T*self.NT_T:].reshape(b,self.D, self.D, self.NT_T, 4).expand(b, self.D, self.D, self.NT_T, 4).contiguous()
            return binary, binary_close, binary_dc, binary_d

        (binary, binary_close, binary_dc, binary_d), unary, root = rules(), terms(), roots()

        return {'unary': unary,
                'root': root,
                'binary': binary,
                'binary_closed': binary_close,
                'binary_dc': binary_dc,
                'binary_d': binary_d,
                'kl': kl(mean, lvar).sum(1)}

    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg._inside(rules=rules, lens=input['seq_len'])
        loss =  (-result['partition'] + rules['kl']).mean()
        return loss


    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            result = self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == 'mbr':
            result = self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

        result['partition'] -= rules['kl']
        return result


