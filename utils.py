import torch
import torch.nn as nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def positional_encoding(seq_len, d_model):
    positions = torch.arange(seq_len)[:, np.newaxis]
    angles = torch.arange(d_model)[np.newaxis, :] / d_model
    angles = angles * np.power(10000, -2 * (angles // 2) / d_model)
    encodings = torch.zeros((seq_len, d_model))
    encodings[:, 0::2] = torch.sin(positions * angles[:, 0::2])
    encodings[:, 1::2] = torch.cos(positions * angles[:, 1::2])
    return encodings.unsqueeze(0).to(device)

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.shape[-1]
    scores = torch.matmul(query, np.swapaxes(key, -2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = torch.where(mask == 0, -1e9, scores)
    attention_weights = softmax(scores)
    return torch.matmul(attention_weights, value), attention_weights

def softmax(x, axis=-1):
    #exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    torch.softmax(x, dim=-1).to(device)
    return torch.softmax(x, dim=-1).to(device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.d_k = d_model 
        self.d_model = d_model
        d_model%num_heads
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.W_v = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        print("value")
        self.W_ff = nn.Linear(d_model, d_model)
        

    
    def forward(self, query, key, value, mask=None):
        #split heads
        b,t,_ = query.size()
        query = query.view(b,t, self.num_heads, self.d_k).transpose(1, 2)
        b,t,_ = key.size()
        key = key.view(b,t, self.num_heads, self.d_k).transpose(1, 2)
        b,t,_ = value.size()
        value = value.view(b,t, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output, _ = scaled_dot_product_attention(query, key, value, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(query.size(0), -1, self.d_model)
        return self.W_ff(attention_output)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.dropout  = nn.Dropout(0.2)  #have to check
        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout1  = nn.Dropout(0.1)  #have to check
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def create_padding_mask(seq):
    return (seq == 0).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.bool)
    return mask