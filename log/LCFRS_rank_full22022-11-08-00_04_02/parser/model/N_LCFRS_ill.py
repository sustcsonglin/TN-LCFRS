import torch
import torch.nn as nn
from parser.modules.res import ResLayer
from ..pcfgs.lcfrs_on5_cpd_ill import PCFG
from ..pcfgs.lcfrs_on5_cpd_masked import PCFG_mask

class NeuralLCFRS_ill(nn.Module):
    def __init__(self, args, dataset):
        super(NeuralLCFRS_ill, self).__init__()
        self.pcfg = PCFG()
        self.pcfg_mask = PCFG_mask()
        self.device = dataset.device
        self.args = args

        self.NT = args.NT
        self.T = args.T
        self.D = args.D
        self.r1 = args.r1
        self.r2 = args.r2
        self.V = len(dataset.word_vocab)

        self.s_dim = args.s_dim

        self.nonterm_emb = nn.Parameter(torch.randn(self.NT + self.T, self.s_dim))
        self.nonterm_emb2 = nn.Parameter(torch.randn((self.NT + self.T)*4, self.s_dim))
        self.d_emb = nn.Parameter(torch.randn(self.D, self.s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))

        self.rc_emb = nn.Parameter(torch.randn(self.s_dim, self.r1))
        self.rd_emb = nn.Parameter(torch.randn(self.s_dim, self.r2))

        self.ratio_c = int(0.9*self.r1)
        self.ratio_c2 = int(0.95*self.r1)
        self.ratio_d = int(0.95*self.r2)

        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.root_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        self.left_c = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.right_c = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.parent_c = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.parent_c2 = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU())
        self.cd_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU() )
        self.cc_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU() )

        self.parent_d = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU() )
        self.left_d = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU() )
        self.left_d_ill = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU() )
        self.right_d = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU() )
        self.right_d_ill = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU() )
        self.dd_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU() )
        self.dc_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim), nn.ReLU() )

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

            head = (self.parent_c(self.nonterm_emb[:self.NT]) @ self.rc_emb).log_softmax(-1)
            head_c1 = head[:, :self.ratio_c].unsqueeze(0).repeat(b,1,1).contiguous()
            head_c2 = head[:, self.ratio_c:self.ratio_c2].unsqueeze(0).repeat(b,1,1).contiguous()
            head_c3 = head[:, self.ratio_c2:].unsqueeze(0).repeat(b,1,1).contiguous()
            left_c = (self.left_c(self.nonterm_emb) @ self.rc_emb[:, :self.ratio_c]).log_softmax(0).unsqueeze(0).repeat(b,1,1).contiguous()
            right_c = (self.right_c(self.nonterm_emb) @ self.rc_emb[:, :self.ratio_c]).log_softmax(0).unsqueeze(0).repeat(b,1,1).contiguous()
            cc = (self.cc_mlp(self.nonterm_emb) @ self.rc_emb[:, self.ratio_c:self.ratio_c2]).log_softmax(0).unsqueeze(0).repeat(b,1,1).contiguous()
            cd = (self.cd_mlp(self.d_emb) @  self.rc_emb[:, self.ratio_c:self.ratio_c2]).log_softmax(0).unsqueeze(0).repeat(b,1,1).contiguous()
            left_d_ill = (self.left_d_ill(self.d_emb) @ self.rc_emb[:, self.ratio_c2:]).log_softmax(0).unsqueeze(0).repeat(b,1,1).contiguous()
            right_d_ill = (self.right_d_ill(self.d_emb) @ self.rc_emb[:, self.ratio_c2:]).log_softmax(0).unsqueeze(0).repeat(b,1,1).contiguous()

            head_d = (self.parent_d(self.d_emb) @ self.rd_emb).log_softmax(-1)
            head_d1 = head_d[:, :self.ratio_d].unsqueeze(0).repeat(b,1,1).contiguous()
            head_d2 = head_d[:, self.ratio_d:].unsqueeze(0).repeat(b,1,1).contiguous()
            left_d = (self.left_d(self.nonterm_emb) @ self.rd_emb[:,:self.ratio_d]).log_softmax(0).unsqueeze(0).repeat(b,1,1).contiguous()
            right_d = (self.right_d(self.nonterm_emb) @ self.rd_emb[:, :self.ratio_d]).log_softmax(0).unsqueeze(0).repeat(b,1,1).contiguous()
            dc = (self.dc_mlp(self.nonterm_emb2) @ self.rd_emb[:, self.ratio_d:]).log_softmax(0).unsqueeze(0).repeat(b,1,1).contiguous()
            dd = (self.dd_mlp(self.d_emb) @ self.rd_emb[:, self.ratio_d:]).log_softmax(0).unsqueeze(0).repeat(b,1,1).contiguous()


            return head_c1, head_c2, head_c3, left_c, right_c, cc, cd, \
                   head_d1, head_d2, left_d, right_d, dc, dd, left_d_ill, right_d_ill

        (head_c1, head_c2, head_c3, left_c, right_c, cc, cd, head_d1, head_d2, left_d, right_d, dc, dd, left_d_ill, right_d_ill), unary, root = rules(), terms(), roots()

        return {'unary': unary,
                'root': root,

                'head_c1': head_c1,
                'head_c2': head_c2,
                'head_c3': head_c3,

                'left_c': left_c,
                'right_c': right_c,
                'cc': cc,
                'cd': cd,

                'head_d1': head_d1,
                'head_d2': head_d2,
                'left_d': left_d,
                'left_d_ill': left_d_ill,
                'right_d': right_d,
                'right_d_ill': right_d_ill,
                'dc': dc,
                'dd': dd,

                'kl': torch.tensor(0, device=self.device)}


    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg._inside(rules=rules, lens=input['seq_len'])
        return -result['partition'].mean()

    def loss_st(self, input, model2):
        rules = self.forward(input)
        with torch.no_grad():
            rules2 = model2.forward(input)
        result = self.pcfg_mask._inside(rules=rules, rules_2= rules2, lens= input['seq_len'])
        return -result['partition'].mean()

    def loss_em(self, input, model2):
        rules = self.forward(input)
        with torch.no_grad():
            rules2 = model2.forward(input)
            head_c1_grd, head_c2_grd, head_d1_grd, head_d2_grd, left_c_grd, right_c_grd, left_d_grd, \
            right_d_grd, cc_grd, cd_grd, dc_grd, dd_grd, unary_grd, gradient_root = self.pcfg.compute_marginals(rules2,input['seq_len'],False)

        loss = (rules['unary'] * unary_grd).sum() + (rules['left_c'] * left_c_grd).sum() \
             + (rules['right_c'] * right_c_grd).sum() + (rules['left_d'] * left_d_grd).sum() \
             + (rules['right_d'] * right_d_grd).sum() + (rules['cc'] * cc_grd).sum() + (rules['dc'] * dc_grd).sum()\
             + (rules['dd'] * dd_grd).sum() + (rules['dc'] * dc_grd).sum() \
             + (rules['root'] * gradient_root).sum() + \
             + (head_d1_grd * rules['head_d1']).sum() + (head_d2_grd * rules['head_d2']).sum() + \
             + (head_c1_grd * rules['head_c1']).sum() + (head_c2_grd * rules['head_c2']).sum()


        loss = -loss/head_c1_grd.shape[0]
        return loss



    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

