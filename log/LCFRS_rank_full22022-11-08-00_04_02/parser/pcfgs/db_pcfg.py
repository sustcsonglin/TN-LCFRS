import pdb

from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import stripe, diagonal_copy_, diagonal, checkpoint
import torch






class PCFG(PCFG_base):



    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):

        terms = rules['unary']
        rule = rules['rule']
        root = rules['root']

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        NTs = slice(0, NT)

        Ts = slice(NT, S)

        X_Y_Zoz = rule[:, :, NTs, :].contiguous()

        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)

        max_depth = 3

        alpha_pred = terms.new_zeros(batch, N, N, NT, S, max_depth).fill_(-1e9)

        alpha_wait = terms.new_zeros(batch, N, N, NT, S, max_depth).fill_(-1e9)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        tmp = contract(terms[:, :, None, :, None] + span_indicator[:, torch.arange(N-1), torch.arange(N-1)+1, None, None, None] + rule[:, None, :, Ts, :], -2)

        alpha_pred[:, torch.arange(N-1), torch.arange(N-1)+1, ...] = tmp.unsqueeze(-1).expand(*tmp.shape, max_depth)

        alpha_wait[:, torch.arange(N-1), torch.arange(N-1)+1, ...] = tmp.unsqueeze(-1).expand(*tmp.shape, max_depth)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ

        @checkpoint
        def PRED(x):
            # x = torch.einsum('bnyd, bxyz -> bnxzd', x, X_Y_Zoz)
            x = contract(x[:, :, None, :, None, :] + X_Y_Zoz[:, None, :, :, :, None], -3)
            return x

        @checkpoint
        def COMP_len1(x, y):
            # x: [b_size, n_span, NT, NT, d]
            # y: [b_size, n_span, NT, NT, d]
            z = contract(x.unsqueeze(-2) + y.unsqueeze(-4), -3)
            return z

        @checkpoint
        def COMP(x, y):
            # x: [b_size, n, w, NT, NT, d]
            # y: [b_size, n, w, NT, NT d]
            # not including the highest depth.
            x = x[..., :-1]
            # not including the lowest depth.
            y = y[..., 1:]
            # [b_size, n, NT, NT, d-1]
            z = contract(x.unsqueeze(-2) + y.unsqueeze(-4), 2)
            z = contract(z, -3)
            z2 = z.new_zeros(*z[..., 0:1].shape, requires_grad=False).fill_(-1e9)
            z = torch.cat([z, z2], dim=-1)
            return z

        for w in range(2, N):
            n = N - w
            Y = stripe(alpha_pred, n, w - 1, (0, 1)).clone()
            Z = stripe(alpha_wait, n, w - 1, (1, w), 0).clone()
            # 平级的.

            if w == N-1:
                x = contract(Y[:, :, -1, :, NT:, :] + terms[:, w - 1:, None, :, None], -2)

            elif w == 2:
                x2 = COMP_len1(Y[:, :, -1, :, :NT, :], Z[:, :, -1])
                # (b, n, NT, depth) -> (b, n, NT, NT, depth)
                x3 = PRED(contract(Y[:, :, -1, :, NT:, :] + terms[:, w-1:, None, :, None], -2) + diagonal(span_indicator, w)[..., None, None])
                x = contract(torch.stack([x2, x3], dim=-1))
                diagonal_copy_(alpha_pred, x, w)
                diagonal_copy_(alpha_wait, x3, w)

            else:
                x1 = COMP(Y[:, :, :-1, :, :NT, ...], Z[:, :, :-1, ...])
                x2 = COMP_len1(Y[:, :, -1, :, :NT], Z[:, :, -1])
                x3 = PRED(contract(Y[:, :, -1, :, NT:, :] + terms[:, w-1:, None, :, None], -2) +  diagonal(span_indicator, w)[..., None, None])
                x = contract(torch.stack([x1, x2, x3], dim=-1), -1)
                diagonal_copy_(alpha_pred, x, w)
                diagonal_copy_(alpha_wait, x3, w)


        logZ = contract(x[..., 0].squeeze(1) + root)



        if viterbi or mbr:
            prediction = self._get_prediction(logZ, span_indicator, lens, mbr=mbr)
            p = []
            for pd in prediction:
                p.append([pd, []])
            return {'partition': logZ,
                    'prediction': p}
        else:
            return {'partition': logZ}


    def _get_prediction(self, logZ, span_indicator, lens, mbr=False):
        batch, seq_len = span_indicator.shape[:2]
        prediction = [[] for _ in range(batch)]
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            assert logZ.requires_grad
            logZ.sum().backward()
            marginals = span_indicator.grad
            if mbr:
                return self._cky_zero_order(marginals.detach(), lens)
            else:
                viterbi_spans = marginals.nonzero().tolist()
                for span in viterbi_spans:
                    #This only contains all left children
                    # we need to induce all the right children.
                    prediction[span[0]].append((span[1], span[2]))

        for i in range(batch):
            prediction[i] = self.left_corner_transform(prediction[i], seq_len-1)

        # self.left_corner_transform()
        return prediction

    def left_corner_transform(self, prediction, L):

        stack = []
        stack.append((0, 1))
        new_prediction = []

        i = 1
        new = (i, i + 1)



        while i < L:
            if new[0] == 0 and new[1] == L:
                break

            if( new in prediction):
                stack.append(new)
                i += 1
                new = (i, i+1)

            else:
                new_prediction.append(new)
                try:
                    assert len(stack) > 0
                except:
                    pdb.set_trace()
                top = stack.pop(-1)
                new = (top[0], new[-1])

                if (new in prediction):
                    stack.append(new)
                    i += 1
                    new = (i, i + 1)

                else:
                    # 说明这是一个right child?
                    while new not in prediction:
                        try:
                            new_prediction.append(new)
                            assert len(stack) > 0
                            top = stack.pop(-1)
                            new = (top[0], new[-1])
                        except:
                            break

            if (0, L) in new_prediction:
                break

        new = prediction + new_prediction
        assert len(new) == (2*L -1)
        return new
