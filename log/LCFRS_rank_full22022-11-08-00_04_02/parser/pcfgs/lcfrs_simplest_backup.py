from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import stripe, diagonal_copy_, diagonal, checkpoint
import torch

import numpy as np


# class PCFG_slow_check(PCFG_base):
#     @torch.enable_grad()
#     def _inside(self, rules, lens, viterbi=False, mbr=False):
#
#         terms = rules['unary']
#         rule = rules['rule']
#         root = rules['root']
#
#         batch, N, T = terms.shape
#         N += 1
#         NT = rule.shape[-1] - T
#         D = rule.shape[1] - NT
#         s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
#         NTs = slice(0, NT)
#         Ts = slice(NT, NT+T)
#         Ds_child = slice(NT+T, NT+T+D)
#         Ds_parent = slice(NT, NT+D)
#
#         X_Y_Z = rule[:, NTs, NTs, NTs].reshape(batch, NT, NT*NT)
#         X_y_Z = rule[:, NTs, Ts, NTs].reshape(batch, NT, NT*T)
#         X_Y_z = rule[:, NTs, NTs, Ts].reshape(batch, NT, NT * T)
#         X_y_z = rule[:, NTs, Ts, Ts].reshape(batch, NT, T * T)
#
#         span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
#
#         def contract(x, dim=-1):
#             if viterbi:
#                 return x.max(dim)[0]
#             else:
#                 return x.logsumexp(dim)
#
#         def kuku(x, y):
#             if viterbi:
#                 return torch.max(x,y)
#             else:
#                 return torch.logaddexp(x, y)
#
#         def contract_m1m2(x):
#             if viterbi:
#                 return  x.max(-1)[0].max(-1)[0]
#
#             else:
#                 return x.logsumexp([-1, -2])
#
#
#         # nonterminals: X Y Z
#         # terminals: x y z
#         # XYZ: X->YZ  XYz: X->Yz  ...
#         @checkpoint
#         def Xyz(y, z,  rule):
#             y_normalizer = y.max(-1)[0]
#             z_normalizer = z.max(-1)[0]
#             y, z = (y-y_normalizer.unsqueeze(-1)).exp(), (z-z_normalizer.unsqueeze(-1)).exp()
#             x = torch.einsum('bny, bnz, bxyz -> bnx', y, z, rule)
#             x = ((x + 1e-9).log() + y_normalizer.unsqueeze(-1) + z_normalizer.unsqueeze(-1))
#             return x
#
#         @checkpoint
#         def XYZ(Y, Z, rule):
#             # n = Y.shape[1]
#             Y = Y[:, :, 1:-1, :]
#             Z = Z[:, :, 1:-1, :]
#             Y_normalizer = Y.max(-1)[0]
#             Z_normalizer = Z.max(-1)[0]
#             Y, Z = (Y-Y_normalizer.unsqueeze(-1)).exp(), (Z-Z_normalizer.unsqueeze(-1)).exp()
#             X = torch.einsum('bnwy, bnwz, bxyz -> bnwx', Y, Z, rule)
#             X = (X + 1e-9).log() + Y_normalizer.unsqueeze(-1) + Z_normalizer.unsqueeze(-1)
#             X = X.logsumexp(2)
#             return X
#
#         @checkpoint
#         def XYz(Y, z, rule):
#             Y = Y[:, :, -1, :]
#             Y_normalizer = Y.max(-1)[0]
#             z_normalizer = z.max(-1)[0]
#             Y, z = (Y-Y_normalizer.unsqueeze(-1)).exp(), (z-z_normalizer.unsqueeze(-1)).exp()
#             X = torch.einsum('bny, bnz, bxyz->bnx', Y, z, rule)
#             X = (X + 1e-9).log() + Y_normalizer.unsqueeze(-1) + z_normalizer.unsqueeze(-1)
#             return X
#
#         @checkpoint
#         def XyZ(y, Z, rule):
#             Z = Z[:, :, 0, :]
#             y_normalizer = y.max(-1)[0]
#             Z_normalizer = Z.max(-1)[0]
#             y, Z = (y-y_normalizer.unsqueeze(-1)).exp(), (Z-Z_normalizer.unsqueeze(-1)).exp()
#             X = torch.einsum('bny, bnz, bxyz-> bnx', y, Z, rule)
#             X = (X + 1e-9).log() + y_normalizer.unsqueeze(-1) + Z_normalizer.unsqueeze(-1)
#             return X
#
#         @checkpoint
#         def merge_disco(Y, Z, z):
#             # 第一种情况.
#             x_1 = (Y[:, :, :-1, :, NTs] + Z[:, :, :-1, None, :]).logsumexp(-1).logsumexp(2)
#             # 第二种情况.
#             x_2 =  (Y[:, :, -1, :, Ts] + z[...,  :]).logsumexp(-1)
#             return torch.stack([x_1, x_2], 0).logsumexp(0)
#
#         @checkpoint
#         def merge_wait(Y, Z):
#             # D, C
#             # A, D
#             # A, C
#             tmp = (Y[:, :, :, None, :, :] + Z[:, :, :,  :, :, None]).logsumexp([2, -2])
#             return tmp
#
#
#         @checkpoint
#         def make_gap(X, rule):
#             return (X[..., None, :, None] + rule[:, None, ...]).logsumexp(-2)
#             X_normalizer = X.max(-1)[0]
#             res = torch.einsum('qnb, qabc -> qnac', (X-X_normalizer[...,None]).exp(), rule)
#
#
#         @checkpoint
#         def make_self_gap(X, rule):
#             X_normalizer = X.max(-1)[0]
#             res = torch.einsum("qnc, qabc -> qnab",  (X - X_normalizer.unsqueeze(-1)).exp(), rule)
#             return (res + 1e-9).log() + X_normalizer[..., None, None]
#
#         for w in range(2, N):
#             n = N - w
#
#             Y_term = terms[:, :n, :, None]
#             Z_term = terms[:, w - 1:, None, :]
#
#             if w == 2:
#                 diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n),
#                                                                torch.arange(n) + w].unsqueeze(-1), w)
#                 continue
#
#             n = N - w
#             x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)
#
#             Y = stripe(s, n, w - 1, (0, 1)).clone()
#             Z = stripe(s, n, w - 1, (1, w), 0).clone()
#
#             if w > 3:
#                 x[0].copy_(XYZ(Y, Z, X_Y_Z))
#
#             x[1].copy_(XYz(Y, Z_term, X_Y_z))
#             x[2].copy_(XyZ(Y_term, Z, X_y_Z))
#
#             x_tmp =  contract(x, dim=0)  + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1)
#             x_tmp2 =  x_tmp.new_zeros(*x_tmp.shape).fill_(-1e9)
#
#             for start in range(n):
#                 ss = x_tmp.new_zeros(x_tmp.shape[0], *x_tmp.shape[2:]).fill_(-1e9)
#                 for split in range(start+1, start+w):
#                     for split2 in range(split+1, start+w):
#                         if split == (start+1):
#                             if split2 == (start+w-1):
#                                 tmp = contract_m1m2(terms[:, split-1, None, :, None] + terms[:, split2, None, None, :] + rule[:, Ds_parent, Ts, Ts])
#                             else:
#                                 tmp = contract_m1m2(terms[:, split-1, None, :, None] + s[:, split2, start+w, None, None, :]  + rule[:, Ds_parent, Ts, NTs])
#                         else:
#                             if split2 == (start + w -1):
#                                 tmp = contract_m1m2(s[:, start, split, None, :, None] + terms[:, split2, None, None, :] + rule[:, Ds_parent, NTs, Ts])
#                             else:
#                                 tmp = contract_m1m2(s[:, start, split, None, :, None] + s[:, split2, start+w, None, None, :] + rule[:, Ds_parent, NTs, NTs])
#
#                         if (split2 - split) > 1:
#                             ss = kuku(ss, contract_m1m2(
#                                             tmp[:,  None, :, None] + s[:, split, split2, None, None, :] + rule[:, NTs,
#                                                                                                          Ds_child,  NTs]))
#                         else:
#                             ss = kuku(ss, contract_m1m2(
#                                             tmp[:,  None, :, None] + terms[:, split, None, None, :] + rule[:, NTs,
#                                                                                                          Ds_child, Ts])
#                                     )
#
#                 x_tmp2[:,start] = ss
#
#             x_tmp = torch.max(x_tmp2, x_tmp)
#
#             diagonal_copy_(s,
#                             x_tmp  ,
#                            w)
#
#         logZ = contract(s[torch.arange(batch), 0, lens] + root)
#         logZ.sum().backward()
#
#         return logZ, span_indicator.grad.nonzero()

# normal space.
class PCFG(PCFG_base):
    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        terms = rules['unary']
        rule = rules['rule'].exp()
        root = rules['root']

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[-1] - T
        D = rule.shape[1] - NT

        # (left + right)
        #
        # left + right +

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        s_make_gap = terms.new_zeros(batch, N, N, D, NT+T).fill_(-1e9)
        s_wait_gap = terms.new_zeros(batch, N, N, NT, D).fill_(-1e9)
        s_wait_C = terms.new_zeros(batch, N, N, NT, NT+T).fill_(-1e9)

        NTs = slice(0, NT)
        Ts = slice(NT, NT+T)
        Ds_child = slice(NT+T, NT+T+D)
        Ds_parent = slice(NT, NT+D)

        X_Y_Z = rule[:, NTs, NTs, NTs]

        D_Y_Z = rule[:, Ds_parent, :NT, :]
        X_D_Z = rule[:, NTs, Ds_child, NTs]
        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        disco_span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)

        if not viterbi:
            s_make_gap[:, torch.arange(N-1), torch.arange(N-1)+1] = (terms[:, :, None, :, None] + (rule[:, None, NT:NT+D, NT:NT+T, :]+1e-9).log()).logsumexp(-2)
            s_wait_gap[:, torch.arange(N-1), torch.arange(N-1)+1] = (terms[:, :, None, None, :] + (rule[:, None, :NT, NT+T:NT+T+D, NT:NT+T]+1e-9).log()).logsumexp(-1)
        else:
            s_make_gap[:, torch.arange(N-1), torch.arange(N-1)+1] = (terms[:, :, None, :, None] + rule[:, None, NT:NT+D, NT:NT+T, :]).max(-2)[0]
            s_wait_gap[:, torch.arange(N-1), torch.arange(N-1)+1] = (terms[:, :, None, None, :] + rule[:, None, :NT, NT+T:NT+T+D, NT:NT+T]).max(-1)[0] +  disco_span_indicator[:, torch.arange(N-1), torch.arange(N-1)+1, None, None]

        X_y_Z = rule[:, NTs, Ts, NTs]
        X_Y_z = rule[:, NTs, NTs, Ts]
        X_y_z = rule[:, NTs, Ts, Ts]

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z,  rule):
            y = y.squeeze(-1)
            z = z.squeeze(-2)
            y_normalizer = y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            y, z = (y-y_normalizer.unsqueeze(-1)).exp(), (z-z_normalizer.unsqueeze(-1)).exp()
            x = torch.einsum('bny, bnz, bxyz -> bnx', y, z, rule)
            x = ((x + 1e-9).log() + y_normalizer.unsqueeze(-1) + z_normalizer.unsqueeze(-1))
            return x

        @checkpoint
        def XYZ(Y, Z, rule):
            # n = Y.shape[1]
            Y = Y[:, :, 1:-1, :]
            Z = Z[:, :, 1:-1, :]
            Y_normalizer = Y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            Y, Z = (Y-Y_normalizer.unsqueeze(-1)).exp(), (Z-Z_normalizer.unsqueeze(-1)).exp()
            X = torch.einsum('bnwy, bnwz, bxyz -> bnwx', Y, Z, rule)
            X = (X + 1e-9).log() + Y_normalizer.unsqueeze(-1) + Z_normalizer.unsqueeze(-1)
            X = X.logsumexp(2)
            return X

        @checkpoint
        def XYz(Y, z, rule):
            Y = Y[:, :, -1, :]
            Y_normalizer = Y.max(-1)[0]
            z = z.squeeze(-2)
            z_normalizer = z.max(-1)[0]

            Y, z = (Y-Y_normalizer.unsqueeze(-1)).exp(), (z-z_normalizer.unsqueeze(-1)).exp()
            X = torch.einsum('bny, bnz, bxyz->bnx', Y, z, rule)
            X = (X + 1e-9).log() + Y_normalizer.unsqueeze(-1) + z_normalizer.unsqueeze(-1)
            return X

        @checkpoint
        def XyZ(y, Z, rule):
            Z = Z[:, :, 0, :]
            y = y.squeeze(-1)
            y_normalizer = y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            y, Z = (y-y_normalizer.unsqueeze(-1)).exp(), (Z-Z_normalizer.unsqueeze(-1)).exp()
            X = torch.einsum('bny, bnz, bxyz-> bnx', y, Z, rule)
            X = (X + 1e-9).log() + y_normalizer.unsqueeze(-1) + Z_normalizer.unsqueeze(-1)
            return X

        @checkpoint
        def merge_disco(Y, Z, z):
            # 第一种情况.
            x_1 = (Y[:, :, :-1, :, NTs] + Z[:, :, :-1, None, :]).logsumexp(-1).logsumexp(2)
            # 第二种情况.
            x_2 =  (Y[:, :, -1, :, Ts] + z[...,  :]).logsumexp(-1)
            return torch.stack([x_1, x_2], 0).logsumexp(0)

        @checkpoint
        def merge_wait(Y, Z):
            tmp = (Y[:, :, :, None, :, :] + Z[:, :, :,  :, :, None]).logsumexp([2, -2])
            return tmp

        @checkpoint
        def make_gap(X, rule):
            # return (X[..., None, :, None] + rule[:, None, ...]).logsumexp(-2)
            X_normalizer = X.max(-1)[0]
            res = torch.einsum('qnb, qabc -> qnac', (X-X_normalizer[...,None]).exp(), rule)
            return (res + 1e-9).log() + X_normalizer[..., None, None]

        @checkpoint
        def make_self_gap(X, rule):
            X_normalizer = X.max(-1)[0]
            res = torch.einsum("qnc, qabc -> qnab",  (X - X_normalizer.unsqueeze(-1)).exp(), rule)
            return (res + 1e-9).log() + X_normalizer[..., None, None]


        for w in range(2, N):
            n = N - w
            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1:, None, :]
            if w == 2:
                x = Xyz(Y_term, Z_term, X_y_z) +  span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1)
                diagonal_copy_(s, x , w)
                if not  viterbi:
                    s_wait_C[:, torch.arange(n), torch.arange(n)+2, ] = (s_make_gap[:, torch.arange(n), torch.arange(n)+1, None, :, :] + s_wait_gap[:, torch.arange(n)+1, torch.arange(n)+2, :, :, None]).logsumexp(-2)
                else:
                    s_wait_C[:, torch.arange(n), torch.arange(n)+2, ] = (s_make_gap[:, torch.arange(n), torch.arange(n)+1, None, :, :] + s_wait_gap[:, torch.arange(n)+1, torch.arange(n)+2, :, :, None]).max(-2)[0]

                x_ = make_gap(x, D_Y_Z)
                diagonal_copy_(s_make_gap , x_, w)

                x_ = make_self_gap(x+disco_span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), X_D_Z)
                diagonal_copy_(s_wait_gap , x_ , w)
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)
            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()
            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))
            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))
            # 从正儿八经的continuous constituents来的.
            tmp_c = contract(x, dim=0)

            Y_d = stripe(s_wait_C, n, w-1, (0,1)).clone()


            tmp_d = merge_disco(Y_d, Z, Z_term)

            if not viterbi:
                x = torch.stack([tmp_c, tmp_d], dim=0).logsumexp(0)
            else:
                x = torch.max(tmp_c, tmp_d)

            x = x + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1)
            diagonal_copy_(s, x, w)
            x_ = make_gap(x, D_Y_Z)
            diagonal_copy_(s_make_gap, x_, w)
            x_ = make_self_gap(x + disco_span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), X_D_Z)
            diagonal_copy_(s_wait_gap, x_, w)
            Y_d = stripe(s_make_gap, n, w-1, (0,1)).clone()
            Z_d = stripe(s_wait_gap, n, w-1, (1,w), 0).clone()
            x_ = merge_wait(Y_d, Z_d)
            diagonal_copy_(s_wait_C, x_ , w)

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        return {'partition': logZ}

    @torch.enable_grad()
    def decode(self, rules, lens, viterbi=False, mbr=False):
        pcfg = PCFG()
        return pcfg._inside(rules=rules, lens=lens, viterbi=viterbi, mbr=mbr)



# class PCFG(PCFG_base):
#
#     @torch.enable_grad()
#     def _inside(self, rules, lens, viterbi=False, mbr=False):
#         terms = rules['unary']
#         rule = rules['rule']
#         root = rules['root']
#
#         batch, N, T = terms.shape
#         N += 1
#         NT = rule.shape[-1] - T
#         D = rule.shape[1] - NT
#         # (left + right
#         # left + right +
#
#         s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
#         s_make_gap = terms.new_zeros(batch, N, N, D, NT+T).fill_(-1e9)
#         s_wait_gap = terms.new_zeros(batch, N, N, NT, D).fill_(-1e9)
#         s_wait_C = terms.new_zeros(batch, N, N, NT, NT+T).fill_(-1e9)
#
#         NTs = slice(0, NT)
#         Ts = slice(NT, NT+T)
#         Ds_child = slice(NT+T, NT+T+D)
#         Ds_parent = slice(NT, NT+D)
#
#         X_Y_Z = rule[:, NTs, NTs, NTs].reshape(batch, NT, NT*NT)
#
#         D_Y_Z = rule[:, Ds_parent, :NT,]
#         X_D_Z = rule[:, NTs, Ds_child, NTs]
#
#         span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
#         disco_span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
#
#         if not viterbi:
#             s_make_gap[:, torch.arange(N-1), torch.arange(N-1)+1] = (terms[:, :, None, :, None] + rule[:, None, NT:NT+D, NT:NT+T, :]).logsumexp(-2)
#             s_wait_gap[:, torch.arange(N-1), torch.arange(N-1)+1] = (terms[:, :, None, None, :] + rule[:, None, :NT, NT+T:NT+T+D, NT:NT+T]).logsumexp(-1)
#         else:
#             s_make_gap[:, torch.arange(N-1), torch.arange(N-1)+1] = (terms[:, :, None, :, None] + rule[:, None, NT:NT+D, NT:NT+T, :]).max(-2)[0]
#             s_wait_gap[:, torch.arange(N-1), torch.arange(N-1)+1] = (terms[:, :, None, None, :] + rule[:, None, :NT, NT+T:NT+T+D, NT:NT+T]).max(-1)[0] +  disco_span_indicator[:, torch.arange(N-1), torch.arange(N-1)+1, None, None]
#
#         X_y_Z = rule[:, NTs, Ts, NTs].reshape(batch, NT, NT*T)
#         X_Y_z = rule[:, NTs, NTs, Ts].reshape(batch, NT, NT * T)
#         X_y_z = rule[:, NTs, Ts, Ts].reshape(batch, NT, T * T)
#
#         def contract(x, dim=-1):
#             if viterbi:
#                 return x.max(dim)[0]
#             else:
#                 return x.logsumexp(dim)
#
#         # nonterminals: X Y Z
#         # terminals: x y z
#         # XYZ: X->YZ  XYz: X->Yz  ...
#         @checkpoint
#         def Xyz(y, z, rule):
#             n = y.shape[1]
#             b_n_yz = (y + z).reshape(batch, n, T * T)
#             b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
#             return b_n_x
#
#         @checkpoint
#         def XYZ(Y, Z, rule):
#             n = Y.shape[1]
#             b_n_yz = contract(Y[:, :, 1:-1, :].unsqueeze(-1) + Z[:, :, 1:-1, :].unsqueeze(-2), dim=2).reshape(batch, n, -1)
#             b_n_x = contract(b_n_yz.unsqueeze(2) + rule.unsqueeze(1))
#             return b_n_x
#
#         @checkpoint
#         def XYz(Y, z, rule):
#             n = Y.shape[1]
#             Y = Y[:, :, -1, :, None]
#             b_n_yz = (Y + z).reshape(batch, n, NT * T)
#             b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
#             return b_n_x
#
#         @checkpoint
#         def merge_disco(Y, Z, z):
#             # 第一种情况.
#             if not viterbi:
#                 x_1 = (Y[:, :, :-1, :, NTs] + Z[:, :, :-1, None, :]).logsumexp(-1).logsumexp(2)
#             # 第二种情况.
#                 x_2 =  (Y[:, :, -1, :, Ts] + z[...,  :]).logsumexp(-1)
#                 return torch.stack([x_1, x_2], 0).logsumexp(0)
#             else:
#                 x_1 = (Y[:, :, :-1, :, NTs] + Z[:, :, :-1, None, :]).max(2)[0].max(-1)[0]
#                 x_2 =  (Y[:, :, -1, :, Ts] + z[...,  :]).max(-1)[0]
#                 return torch.max(x_1, x_2)
#
#         @checkpoint
#         def merge_wait(Y, Z):
#             # D, C
#             # A, D
#             # A, C
#             if viterbi:
#                 tmp = (Y[:, :, :, None, :, :] + Z[:, :, :,  :, :, None]).max(2)[0].max(-2)[0]
#             else:
#                 tmp = (Y[:, :, :, None, :, :] + Z[:, :, :,  :, :, None]).logsumexp([2, -2])
#             return tmp
#
#         @checkpoint
#         def XyZ(y, Z, rule):
#             n = Z.shape[1]
#             Z = Z[:, :, 0, None, :]
#             b_n_yz = (y + Z).reshape(batch, n, NT * T)
#             b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
#             return b_n_x
#
#         @checkpoint
#         def make_gap(X, rule):
#             if not viterbi:
#                 return (X[..., None, :, None] + rule[:, None, ...]).logsumexp(-2)
#             else:
#                 return (X[..., None, :, None] + rule[:, None, ...]).max(-2)[0]
#
#         @checkpoint
#         def make_self_gap(X, rule):
#             if not viterbi:
#                 return (X[..., None, None, :] + rule[:, None, ...]).logsumexp(-1)
#             else:
#                 return (X[..., None, None, :] + rule[:, None, ...]).max(-1)[0]
#
#         for w in range(2, N):
#             n = N - w
#             Y_term = terms[:, :n, :, None]
#             Z_term = terms[:, w - 1:, None, :]
#             if w == 2:
#                 x = Xyz(Y_term, Z_term, X_y_z) +  span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1)
#                 diagonal_copy_(s, x , w)
#                 if not  viterbi:
#                     s_wait_C[:, torch.arange(n), torch.arange(n)+2, ] = (s_make_gap[:, torch.arange(n), torch.arange(n)+1, None, :, :] + s_wait_gap[:, torch.arange(n)+1, torch.arange(n)+2, :, :, None]).logsumexp(-2)
#                 else:
#                     s_wait_C[:, torch.arange(n), torch.arange(n)+2, ] = (s_make_gap[:, torch.arange(n), torch.arange(n)+1, None, :, :] + s_wait_gap[:, torch.arange(n)+1, torch.arange(n)+2, :, :, None]).max(-2)[0]
#
#                 x_ = make_gap(x, D_Y_Z)
#                 diagonal_copy_(s_make_gap , x_, w)
#
#                 x_ = make_self_gap(x+disco_span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), X_D_Z)
#                 diagonal_copy_(s_wait_gap , x_ , w)
#                 continue
#
#             n = N - w
#             x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)
#             Y = stripe(s, n, w - 1, (0, 1)).clone()
#             Z = stripe(s, n, w - 1, (1, w), 0).clone()
#             if w > 3:
#                 x[0].copy_(XYZ(Y, Z, X_Y_Z))
#             x[1].copy_(XYz(Y, Z_term, X_Y_z))
#             x[2].copy_(XyZ(Y_term, Z, X_y_Z))
#             # 从正儿八经的continuous constituents来的.
#             tmp_c = contract(x, dim=0)
#
#             Y_d = stripe(s_wait_C, n, w-1, (0,1)).clone()
#
#
#             tmp_d = merge_disco(Y_d, Z, Z_term)
#
#             if not viterbi:
#                 x = torch.stack([tmp_c, tmp_d], dim=0).logsumexp(0)
#             else:
#                 x = torch.max(tmp_c, tmp_d)
#
#             x = x + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1)
#             diagonal_copy_(s, x, w)
#
#             x_ = make_gap(x, D_Y_Z)
#             diagonal_copy_(s_make_gap, x_, w)
#
#             x_ = make_self_gap(x
#                                + disco_span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), X_D_Z)
#             diagonal_copy_(s_wait_gap, x_, w)
#             Y_d = stripe(s_make_gap, n, w-1, (0,1)).clone()
#             Z_d = stripe(s_wait_gap, n, w-1, (1,w), 0).clone()
#             x_ = merge_wait(Y_d, Z_d)
#             diagonal_copy_(s_wait_C, x_ , w)
#
#         logZ = contract(s[torch.arange(batch), 0, lens] + root)
#
#         if viterbi or mbr:
#             prediction = [[[], []] for _ in range(batch)]
#             # to avoid some trivial corner cases.
#             if lens.max() >= 3:
#                 assert logZ.requires_grad
#                 logZ.sum().backward()
#                 continuous_span_indicator = span_indicator.grad
#                 gap_indicator = disco_span_indicator.grad
#
#                 viterbi_spans = continuous_span_indicator.nonzero().tolist()
#                 for span in viterbi_spans:
#                     prediction[span[0]][0].append((span[1], span[2]))
#
#                 gaps = gap_indicator.nonzero().tolist()
#                 continuous_span_indicator[:, 0, -1] = 1
#                 for span in gaps:
#                     b_idx = span[0]
#                     start = span[1]
#                     end = span[2]
#                     left = start - 1
#                     right = end + 1
#                     while True:
#                         if continuous_span_indicator[b_idx][left][right] > 0:
#                             break
#                         if left-1>=0 and continuous_span_indicator[b_idx][left-1][right] > 0:
#                             left -= 1
#                             break
#                         if right+1 < continuous_span_indicator.shape[1] and continuous_span_indicator[b_idx][left][right+1]>0:
#                             right+=1
#                             break
#                         if left-1 >= 0:
#                             left-=1
#                         if right+1 < continuous_span_indicator.shape[1]:
#                             right+=1
#                     assert continuous_span_indicator[b_idx][left][right] > 0
#                     prediction[b_idx][1].append((left,start, end, right))
#
#
#             return  {
#                 'partition': logZ,
#                 'prediction': prediction
#             }
#
#         else:
#             return {'partition': logZ}


#
#
# if __name__ == '__main__':
#     NT = 5
#     T = 10
#     D = 5
#     B = 3
#     n = 6
#     binary = torch.rand(B, NT+D, NT+D+T, NT+T)
#     unary = torch.rand(B, n, T)
#     root = torch.rand(B, NT)
#     lens = torch.zeros(B).fill_(n).long()
#
#     rule = {'rule': binary,
#             'unary': unary,
#             'root': root}
#     pcfg = Faster_PCFG()
#     res = pcfg._inside(rule, lens)
#     print(res)
#     pcfg2 = PCFG()
#     res = pcfg2._inside(rule, lens)
#     print(res)
#
#
#
