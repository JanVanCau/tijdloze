import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.d_model = d_model              
        self.vocab_size = vocab_size        
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)    #scaling the embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # Ensure pe isn't a learnable parameter during training
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)] # Add the positional embeddings to the token embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
       
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.query_linear = nn.Linear(d_model, d_model, bias=False)  #WQ-matrix
        self.key_linear = nn.Linear(d_model, d_model, bias=False)   #WK-matrix
        self.value_linear = nn.Linear(d_model, d_model, bias=False)  #WV-matrix 
        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        seq_length = x.size(1)
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)     # Split input embeddings and permute
        return x.permute(0, 2, 1, 3)

    def compute_attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  #QxK(transposed) / sqrt(head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)                #softmax
        return torch.matmul(attention_weights, value)                # x V

    def combine_heads(self, x, batch_size):
        seq_length = x.size(2)                                      #DataCamp -> seq_length = x.size(1)
        x = x.permute(0, 2, 1, 3).contiguous()                      # Combine the outputs back together (= concatenate)
        return x.reshape(batch_size, seq_length, self.d_model)             

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Build the forward pass
        query = self.split_heads(self.query_linear(x), batch_size)  #Q
        key = self.split_heads(self.key_linear(x), batch_size)  #K 
        value = self.split_heads(self.value_linear(x), batch_size)  #V 
        
        attention_weights = self.compute_attention(query, key, value, mask)
        output = self.combine_heads(attention_weights, batch_size)
        return self.output_linear(output)


class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))         # Apply dropout and the first layer normalization
        ff_output = self.ff_sublayer(x)
        x = self.norm2(x + self.dropout(ff_output))           # Apply dropout and the second layer normalization
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.final = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.final(x)
