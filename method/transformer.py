import torch
import math
from torch.nn.functional import softmax


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim_emb, max_len):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, dim_emb)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_emb, 2) * -(math.log(10000.0) / dim_emb))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).cuda()

    def forward(self, x):
        return x + self.encoding[:, :x.shape[1], :]


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_emb, dim_head, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dim_model = self.dim_head * self.num_heads
        self.fc_q = torch.nn.Linear(dim_emb, self.dim_model)
        self.fc_k = torch.nn.Linear(dim_emb, self.dim_model)
        self.fc_v = torch.nn.Linear(dim_emb, self.dim_model)
        self.fc_out = torch.nn.Linear(self.dim_model, dim_emb)

    def forward(self, q, k, v):
        # x: (num_batch, len_seq, dim_emb)
        # q, k, v: (num_batch, num_heads, len_seq, dim_head)
        # attn: (num_batch, num_heads, len_seq, len_seq)
        # out: (num_batch, len_seq, dim_emb)

        # Transform the input query, key, and value.
        q = self.fc_q(q).view(q.shape[0], -1, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.fc_k(k).view(k.shape[0], -1, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.fc_v(v).view(v.shape[0], -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Calculate self-attentions.
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.dim_head)
        attn = softmax(attn, dim=3)

        # Generate new latent embeddings of the value based on the calculated attentions.
        h = torch.matmul(attn, v).transpose(1, 2).contiguous().view(q.shape[0], -1, self.dim_model)
        out = self.fc_out(h)

        return out


class TransformerBlock(torch.nn.Module):
    def __init__(self, dim_emb, dim_head, num_heads, len_spect):
        super(TransformerBlock, self).__init__()
        self.pos_encoder = PositionalEncoding(dim_emb, len_spect)
        self.attn_layer = MultiHeadAttention(dim_emb, dim_head, num_heads)
        self.ff_layer = torch.nn.Sequential(
            torch.nn.Linear(dim_emb, dim_emb),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim_emb, dim_emb)
        )
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        z = self.dropout(self.pos_encoder(x))
        z = self.dropout(z + self.attn_layer(z, z, z))
        # z = self.dropout(z + self.attn_layer(z, z, z))
        # out = self.dropout(z + self.ff_layer(z))
        out = self.ff_layer(z)

        return out
