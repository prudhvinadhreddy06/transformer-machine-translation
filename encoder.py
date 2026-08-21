import torch
import torch.nn as nn
import math
from utils import MultiHeadAttention, PositionWiseFeedForward, positional_encoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout) # check
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attention(x, x, x, mask)
        drop_attn = self.dropout(attn_output)
        x = self.norm1(x + drop_attn)
        ff_output = self.feed_forward(x)
        drop_ff = self.dropout(ff_output)
        return self.norm2(x + drop_ff)

    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout1 = nn.Dropout(dropout) # check
        self.pos_encoding = positional_encoding(max_seq_length, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        seq_length = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        

        pos_encoding = self.pos_encoding[:, :seq_length, :].to(x.device)
        
        x = x + pos_encoding
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x