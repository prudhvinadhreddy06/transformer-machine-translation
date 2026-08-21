import torch
import torch.nn as nn
import math
from utils import MultiHeadAttention, PositionWiseFeedForward, positional_encoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        self.dropout1 = nn.Dropout(dropout)  #check
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(dropout)  #check
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)  #check
        self.norm3 = nn.LayerNorm(d_model)
        

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attention(x, x, x, tgt_mask)
        drop = self.dropout(attn_output)
        x = self.norm1(x + drop)
        attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        cross_drop = self.dropout(attn_output)
        x = self.norm2(x + cross_drop)
        ff_output = self.feed_forward(x)
        return  self.norm3(x + self.dropout(ff_output))
        

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout1 = nn.Dropout(dropout) #check
        self.positional_encoding = positional_encoding(max_seq_length, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        

    def forward(self, x, enc_output, src_mask, tgt_mask):
        seq_length = x.size(1)
        sqrt_d_model = math.sqrt(self.d_model)
        x = self.embedding(x) * sqrt_d_model
        pos_enc = self.positional_encoding[:, :seq_length, :].to(x.device)
        x = x + pos_enc
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x.to(device)