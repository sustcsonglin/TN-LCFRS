import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# copied from: https://github.com/lifengjin/charInduction/blob/9f17530ad949a8085eda454cc84faafe2e7fd75b/char_coding_models.py#L16

class CharProbRNN(nn.Module):
    def __init__(self, num_chars, state_dim=256, hidden_size=512, num_layers=4, dropout=0.2):
        super(CharProbRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.state_dim = state_dim

        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        # self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        self.top_fc = nn.Linear(hidden_size, num_chars)
        self.char_embs = nn.Embedding(num_chars, hidden_size)

        # self.cat_emb_expansion = nn.Sequential(nn.Linear(state_dim, hidden_size), nn.ReLU())
        self.cat_emb_expansion = nn.Sequential(nn.Linear(state_dim, hidden_size*num_layers), nn.ReLU())
        self.cat_emb_expansion2 = nn.Sequential(nn.Linear(state_dim, hidden_size*num_layers), nn.ReLU())

        torch.nn.init.kaiming_normal_(self.char_embs.weight.data)

        self.bos_idx = 2
        self.eos_idx = 3
        self.num_char = num_chars

    def forward(self, x, cat_embs):
        '''
        n_token_batch: the total number of tokens within a batch
        n_char_batch: the total number of characters within a batch
        pt_num: the number of preterminals.
        '''
        pt_num = cat_embs.shape[0]

        mask = x.ne(0)

        mask_src = mask & x.ne(self.eos_idx)
        mask_tgt = mask & x.ne(self.bos_idx)

        lens = mask.sum(-1)
        char_mask = (lens > 0)

        # (n_token_batch, fix_len, char_emb_size)
        char_emb = self.char_embs(x[char_mask])

        # (n_token_batch * pt_num, fix_len, char_emb_size)
        char_emb = char_emb.unsqueeze(0).expand(pt_num, -1, -1, -1).reshape(-1, char_emb.shape[1], char_emb.shape[2])

        l = lens[char_mask]
        n = l.shape[0]
        l = l.unsqueeze(0).expand(pt_num, -1).reshape(-1, 1).squeeze(-1)

        char_emb = pack_padded_sequence(char_emb, l.cpu(), batch_first=True, enforce_sorted=False)

        # (pt_num, hidden_size, rnn_layer_num)
        init_c0 = self.cat_emb_expansion(cat_embs).reshape(cat_embs.shape[0], self.hidden_size, -1)
        # (n_token_batch, p_num; hidden; Layer*D) -> (layer*D, p_num*n_token_batch, hidden)
        init_c0 = init_c0.unsqueeze(1).expand(pt_num, n, init_c0.shape[1], init_c0.shape[2]).reshape(-1, init_c0.shape[1], init_c0.shape[2]).permute(2,0,1)
        # init_h0 = torch.zeros_like(init_c0)
        # # (pt_num, hidden_size, rnn_layer_num)
        # init_h0 = self.cat_emb_expansion2(cat_embs).reshape(cat_embs.shape[0], self.hidden_size, -1)
        # (n_token_batch, p_num; hidden; Layer*D) -> (layer*D, p_num*n_token_batch, hidden)
        # init_h0 = init_h0.unsqueeze(1).expand(pt_num, n, init_h0.shape[1], init_h0.shape[2]).reshape(-1, init_h0.shape[1], init_h0.shape[2]).permute(2,0,1)

        output, _ = self.rnn(char_emb, (init_c0, init_c0))

        # (n_token_batch * p_num, fix_len, hidden_state)
        output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        # (p_num, n_token_batch,  fix_len, hidden_state)
        # output = output.reshape(pt_num, n, x.shape[2], self.hidden_size)

        char_mask_extended = char_mask.unsqueeze(0).expand(pt_num, -1, -1)

        # (p_num, batch, len, fix_len, hidden_state)
        output = output.new_zeros(pt_num, x.shape[0], x.shape[1], x.shape[2], self.hidden_size).masked_scatter_(char_mask_extended.unsqueeze(-1).unsqueeze(-1), output)

        mask_src_expanded = mask_src.unsqueeze(0).expand(pt_num, -1, -1, -1)
        mask_tgt_expanded = mask_tgt.unsqueeze(0).expand(pt_num, -1, -1, -1)

        # (p_num * n_char_batch, hidden_state)
        output = output[mask_src_expanded]
        output = self.top_fc(output).log_softmax(-1)

        x_expanded = x.unsqueeze(0).expand(pt_num, -1, -1, -1)
        x_tgt = x_expanded[mask_tgt_expanded]

        logits = output.gather(-1, x_tgt.unsqueeze(-1)).squeeze(-1)
        final = logits.new_zeros(pt_num, x.shape[0], x.shape[1], x.shape[2]).masked_scatter_(mask_tgt_expanded, logits)
        final = final.sum(-1).permute(1, 2, 0)
        # return: (batch, L, pt_num)
        # res2 = self.forward2(x, cat_embs)
        # assert torch.isclose(res2, final).all()
        return final.contiguous()

    def prep_input(self, chars, cat_embs):
        # cat_embs is num_cat, cat_dim
        # chars is num_words, word/char_tensor
        embeddings = []
        for i in range(len(chars)):
            for j in range(len(chars[i])):
                embeddings.append(self.char_embs.forward(torch.tensor(chars[i][j][:-1], device=cat_embs.device, dtype=torch.long))) # no word end token
        packed_char_embs = nn.utils.rnn.pack_sequence(embeddings, enforce_sorted=False) # len, batch, embs
        expanded_cat_embs = cat_embs.unsqueeze(1).expand(-1, packed_char_embs.data.size(0), -1) # numcat,batch, catdim
        return packed_char_embs, expanded_cat_embs



    def forward2(self, chars, cat_embs, set_pcfg=True): # does not use set pcfg
        char_embs, cat_embs = self.prep_input(chars, cat_embs)
        Hs = []
        lens = 0
        for cat_tensor in cat_embs: # each cat at one time
            # for simple RNNs
            # # cat_tensor is batch, dim
            # cat_tensor = cat_tensor.unsqueeze(0).expand(self.num_layers, -1, -1)
            # cat_tensor = self.cat_emb_expansion(cat_tensor)
            # all_hs, _ = self.rnn.forward(char_embs, cat_tensor)
            # all_hs = nn.utils.rnn.pad_packed_sequence(all_hs) # len, batch, embs
            # Hs.append(all_hs[0].transpose(0,1))
            # lens = all_hs[1]

            # for LSTMs with 3d linears
            cat_tensor = self.cat_emb_expansion(cat_tensor) # batch, hidden*numlayers
            cat_tensor = cat_tensor.reshape(cat_tensor.shape[0], self.hidden_size, -1)
            cat_tensor = cat_tensor.permute(2, 0, 1)
            h0_tensor = torch.zeros_like(cat_tensor)
            all_hs, _ = self.rnn.forward(char_embs, (h0_tensor, cat_tensor))
            all_hs = nn.utils.rnn.pad_packed_sequence(all_hs) # len, batch, embs
            Hs.append(all_hs[0].transpose(0,1))
            lens = all_hs[1]

        Hses = torch.stack(Hs, 0)
        # Hses = nn.functional.relu(Hses)
        scores = self.top_fc.forward(Hses) # cats, batch, num_chars_in_word, num_chars
        logprobs = torch.nn.functional.log_softmax(scores, dim=-1)
        total_logprobs = []

        for idx, length in enumerate(lens.tolist()):
            this_word_logprobs = logprobs[:, idx, :length, :] # cats, (batch_scalar), num_chars_in_word, num_chars
            sent_id = idx // len(chars[0])
            word_id = idx % len(chars[0])
            targets = chars[sent_id][word_id][1:]
            this_word_logprobs = this_word_logprobs[:, range(this_word_logprobs.shape[1]), targets]  # cats, num_chars_in_word
            total_logprobs.append(this_word_logprobs.sum(-1)) # cats
        total_logprobs = torch.stack(total_logprobs, dim=0) # batch, cats
        total_logprobs = total_logprobs.reshape(len(chars), -1, total_logprobs.shape[1]) # sentbatch, wordbatch, cats
        # total_logprobs = total_logprobs.transpose(0, 1) # wordbatch, sentbatch, cats
        return total_logprobs
