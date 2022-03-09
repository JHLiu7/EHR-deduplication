import torch, math
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.conv import Conv1d

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextBaseModule(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout) -> None:
        super(TextBaseModule, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.word_dropout = nn.Dropout(dropout)

    def init_embeddings(self, embeddings):
        """
        embeddings: np.array
        """
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings))

    def freeze_embeddings(self, freeze=False):
        self.embed.weight.requires_grad = not freeze


class AttentionWithContext(nn.Module):
    def __init__(self, input_size, attention_size, context_dim=1):
        super(AttentionWithContext, self).__init__()
        
        self.linear = nn.Linear(input_size, attention_size)
        self.context= nn.Linear(attention_size, context_dim)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.context.weight)
        
    def _masked_softmax(self, att, seq_len):
        """
        att: (num_seq, pad_seq_dim)
        seq_len: (num_seq,)
        """
        index = torch.arange(0, int(seq_len.max())).unsqueeze(1).type_as(att)
        seq_len = seq_len.type_as(att)
        # print(index, seq_len)

        mask = (index < seq_len.unsqueeze(0))

        score = torch.exp(att) * mask.transpose(0, 1)
        dist = score / torch.sum(score, dim=-1, keepdim=True)

        return dist
        
    def forward(self, seq_enc, seq_len=None):
        
        att = torch.tanh(self.linear(seq_enc))
        att = self.context(att).squeeze(-1)
        
        if seq_len is not None:
            score = self._masked_softmax(att, seq_len)
        else:
            score = torch.softmax(att, dim=-1)
        # return score
        enc_weighted = score.unsqueeze(-1) * seq_enc
        
        return enc_weighted.sum(1), score


##############################################################################################################
## layers for medium length context

# modified from https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network
# replaced label-wise attention with standard attention pooling 

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(math.floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


class MultiResCNN(TextBaseModule):
    def __init__(self, Y, filter_num, filter_sizes, num_layers, vocab_size, embed_dim, dropout, **kwargs):
        super().__init__(vocab_size, embed_dim, dropout)
        # single layer 
        
        self.convs = nn.ModuleList()

        kernels = filter_sizes.split(',')
        self.encode_dim = filter_num * len(kernels)

        self.conv_dict = {1: [embed_dim, filter_num],
                     2: [embed_dim, 100, filter_num],
                     3: [embed_dim, 150, 100, filter_num],
                     4: [embed_dim, 200, 150, 100, filter_num]
                     }

        for k in kernels:
            one_channel = nn.ModuleList()
            cnn = nn.Conv1d(embed_dim, embed_dim, kernel_size=int(k),
                padding=int(math.floor(int(k)/2)) )
            nn.init.xavier_uniform_(cnn.weight)
            one_channel.add_module('baseconv', cnn)

            conv_dim = self.conv_dict[num_layers]
            for idx in range(num_layers):
                res = ResidualBlock(conv_dim[idx], conv_dim[idx+1], int(k), 1, True, dropout)
                one_channel.add_module('resconv-{}'.format(idx), res)

            self.convs.add_module('channel-{}'.format(k), one_channel)

        self.attention = AttentionWithContext(self.encode_dim, self.encode_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.embed(x)
        x = self.word_dropout(x)
        x = x.transpose(1,2)

        conv_result = []
        for conv in self.convs:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)  # (N, W, F)
            conv_result.append(tmp)

        x = torch.cat(conv_result, dim=2) # (N, W, nF)
        out, _ = self.attention(x)

        return self.dropout(out)

##############################################################################################################

## layers for full length context / hierarchical modeling

def order_notes(rep_notes, stay_order):
    """
    stay_order: (num_stay, padded_note_len)
    """
    
    rep_notes = F.pad(rep_notes, (0,0,1,0))
    notes = rep_notes[stay_order.view(-1)]
    notes = notes.view(stay_order.size(0), stay_order.size(1), notes.size(-1))
    
    return notes

class TokenHANEncoder(TextBaseModule):
    def __init__(self, vocab_size, embed_dim, dropout):
        super(TokenHANEncoder, self).__init__(vocab_size, embed_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def _reduce_step(self, encoder, attention, sequence, seq_lengths, pack_seq=True):
        if pack_seq: # sort note order by lengths for packing
            seq_lengths, seq_perm_idx = seq_lengths.sort(dim=0, descending=True)
            sequence = sequence[seq_perm_idx]
            sequence = pack_padded_sequence(sequence, lengths=seq_lengths.tolist(), batch_first=True)

        if encoder.__str__().startswith('Conv'):
            sequence = sequence.transpose(1, 2)
            enc_seq = encoder(sequence).transpose(1, 2)
        else:
            enc_seq, _ = encoder(sequence)

        if pack_seq: # pad seq to calc attention
            enc_seq, seq_lengths = pad_packed_sequence(enc_seq, batch_first=True)
        
        ft_seq, attn_weight = attention(enc_seq, seq_lengths)

        if pack_seq: # unsort seq
            _, seq_unperm_idx = seq_perm_idx.sort(dim=0, descending=False)
            ft_seq = ft_seq[seq_unperm_idx]
            attn_weight = attn_weight[seq_unperm_idx]

        ft_seq = self.dropout(ft_seq)

        return ft_seq, attn_weight 

    def forward(self, notes, note_lengths, stay_lengths, stay_order, pack_seq=True):
        """
        notes: (num_notes, padded_note_len)
        note_lengths: (num_notes)
        stay_lengths: (num_stays) # each stay has a number of notes
        stay_order: (num_stays, padded_stay_len)
        """

        notes = self.embed(notes)
        notes = self.word_dropout(notes) # (num_note, len_note, dim)

        # pool note ft 
        ft_note, attn_note = self._reduce_step(
            self.note_encoder, self.note_attention,
            notes, note_lengths, pack_seq=pack_seq
        )

        stays = order_notes(ft_note, stay_order)

        # pool stay ft
        ft_stay, attn_stay = self._reduce_step(
            self.stay_encoder, self.stay_attention,
            stays, stay_lengths, pack_seq=pack_seq
        )

        return ft_stay, (attn_note, attn_stay)

class Token_GRU_HAN(TokenHANEncoder):
    def __init__(self, hidden_state, num_layers, vocab_size, embed_dim, dropout, **kwargs) -> None:
        super(Token_GRU_HAN, self).__init__(vocab_size, embed_dim, dropout)

        self.note_encoder = nn.GRU(embed_dim, hidden_state, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.stay_encoder = nn.GRU(hidden_state*2, hidden_state, bidirectional=True, batch_first=True)

        self.note_attention = AttentionWithContext(hidden_state*2, hidden_state*2)
        self.stay_attention = AttentionWithContext(hidden_state*2, hidden_state*2)

        self.encode_dim = hidden_state*2


class ConvEncoder(nn.Module):
    def __init__(self, filter_num, filter_sizes, embed_dim):
        super().__init__()

        kernels = [int(k) for k in filter_sizes.split(',')]

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, filter_num, kernel_size=k, padding=int(math.floor( k / 2)))
            for k in kernels
        ])

    def forward(self, x):
        # (b, dim, seq)
        out = torch.cat([conv(x) for conv in self.convs], 1) # (b, fxn, seq)
        return out

class Token_CNN_HAN(TokenHANEncoder):
    def __init__(self, filter_num, filter_sizes, vocab_size, embed_dim, dropout, **kwargs) -> None:
        super(Token_CNN_HAN, self).__init__(vocab_size, embed_dim, dropout)
        """
        filter num: int
        filter sizes: '3,5' or '3,5-3' for each of two layers
        """
        if '-' in filter_sizes:
            filter_size1, filter_size2 = filter_sizes.split('-')
        else:
            filter_size1 = filter_sizes
            filter_size2 = filter_sizes

        self.note_encoder = ConvEncoder(filter_num, filter_size1, embed_dim)
        self.note_dim = len(filter_size1.split(',')) * filter_num

        if len(filter_size2) > 1:
            self.stay_encoder = ConvEncoder(filter_num, filter_size2, self.note_dim)
        else:
            k = int(filter_size2)
            p = int(math.floor( k / 2))
            self.stay_encoder = Conv1d(self.note_dim, filter_num, kernel_size=k, padding=p)
        self.stay_dim = len(filter_size2.split(',')) * filter_num

        self.note_attention = AttentionWithContext(self.note_dim, self.note_dim)
        self.stay_attention = AttentionWithContext(self.stay_dim, self.stay_dim)

        self.encode_dim = self.stay_dim


##############################################################################################################
## classifiers


class HANModelRNN(nn.Module):
    def __init__(self, Y, hidden_state, num_layers, vocab_size, embed_dim, dropout, **kwargs):
        super().__init__()

        self.encoder = Token_GRU_HAN(hidden_state, num_layers, vocab_size, embed_dim, dropout)
        self.fc = nn.Linear(self.encoder.encode_dim, Y)
        self.pack_seq = True
        self.Y = Y

    def forward(self, notes, note_lengths, stay_lengths, stay_order):
        out, _ = self.encoder(notes, note_lengths, stay_lengths, stay_order, self.pack_seq)
        
        y_hat = self.fc(out)
        if self.Y == 1:
            y_hat = torch.relu(y_hat).squeeze()
        return y_hat


class HANModelCNN(nn.Module):
    def __init__(self, Y, filter_num, filter_sizes, vocab_size, embed_dim, dropout, **kwargs):
        super().__init__()

        self.encoder = Token_CNN_HAN(filter_num, filter_sizes, vocab_size, embed_dim, dropout)
        self.fc = nn.Linear(self.encoder.encode_dim, Y)
        self.pack_seq = False
        self.Y = Y

    def forward(self, notes, note_lengths, stay_lengths, stay_order):
        out, _ = self.encoder(notes, note_lengths, stay_lengths, stay_order, self.pack_seq)
        
        y_hat = self.fc(out)
        if self.Y == 1:
            y_hat = torch.relu(y_hat).squeeze()
        return y_hat


class FlatModel(nn.Module):
    def __init__(self, Y, filter_num, filter_sizes, num_layers, vocab_size, embed_dim, dropout, **kwargs):
        super().__init__()

        self.encoder = MultiResCNN(Y, filter_num, filter_sizes, num_layers, vocab_size, embed_dim, dropout)
        # self.encoder = TextCNN(filter_num, filter_sizes, vocab_size, embed_dim, dropout)
        self.fc = nn.Linear(self.encoder.encode_dim, Y)

        self.Y = Y

    def forward(self, x):
        out = self.encoder(x)
        
        y_hat = self.fc(out)
        if self.Y == 1:
            y_hat = torch.relu(y_hat).squeeze()
        return y_hat