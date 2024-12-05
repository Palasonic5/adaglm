import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_neighbor, use_bias):
        super().__init__()
        
        self.d_model = out_dim
        self.d_k = out_dim // num_heads
        
        self.num_heads = num_heads
        self.num_neighbor = num_neighbor
        self.out_dim = out_dim
        
        self.Q = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim, bias=use_bias)

    def forward(self, sequence):
        # sequence should be in shape [batch_size, 1 + num_neighbor, in_dim]
        batch_size = sequence.shape[0]
        
        Q_h = self.Q(sequence).view(batch_size, 1 + self.num_neighbor, self.num_heads, self.d_k).transpose(1,2) # [b, head, 1+neib, out]
        K_h = self.K(sequence).view(batch_size, 1 + self.num_neighbor, self.num_heads, self.d_k).transpose(1,2)
        V_h = self.V(sequence).view(batch_size, 1 + self.num_neighbor, self.num_heads, self.d_k).transpose(1,2)

        attention_scores = torch.matmul(Q_h, K_h.transpose(2,3)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))  #[b, head, 1+neib, 1+neib]
        attention_scores = F.softmax(attention_scores, dim=-1) #[b, head, 1+neib, 1+neib]
        weighted_values = torch.matmul(attention_scores, V_h) #[b, head, 1+neib, out]
        # output = weighted_values.mean(dim = 1)  # take mean of each head -> [b, 1+neib, out]
        output = weighted_values.transpose(1, 2).contiguous().view(batch_size, 1+self.num_neighbor, self.out_dim)

        # print('gtf layer output shape', output.shape)

        return output

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, num_neighbor, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        # self.seq_length = num_neighbor + 1
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim, num_heads, num_neighbor, use_bias)
        
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, sequence):
        h_in1 = sequence # for first residual connection
        # print('hin1 shape:', h_in1.shape)

        h = self.attention(sequence)
        # print('attention out shape:', attn_out.shape)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        # print('o shape:', h.shape)
        
        if self.residual:
            h = h_in1 + h 
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h.transpose(1,2)).transpose(1,2)
        
        h_in2 = h 
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h.transpose(1,2)).transpose(1,2)       
        # print('attention output', h.shape)

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
