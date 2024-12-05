import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_neighbor, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_neighbor = num_neighbor
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def forward(self, center_embedding, neighbor_embeddings):

        batch_size = center_embedding.shape[0]
        
        Q_h = self.Q(center_embedding).view(batch_size, 1, self.out_dim, self.num_heads)
        K_h = self.K(neighbor_embeddings).view(batch_size, self.num_neighbor, self.out_dim, self.num_heads)
        V_h = self.V(neighbor_embeddings).view(batch_size, self.num_neighbor, self.out_dim, self.num_heads)

        # Adjust Q_h by repeating it for each neighbor to match K_h's batch size
        Q_h = Q_h.repeat(1, self.num_neighbor, 1, 1)  # Now Q_h is [b, num_neighbor, out_dim, num_heads]

        # Calculate attention scores using dot product attention mechanism
        attention_scores = torch.matmul(Q_h, K_h.transpose(2,3)) / np.sqrt(self.out_dim)  #[b, neib, hid, head] * [b, neib, head, hid]
        attention_scores = F.softmax(attention_scores, dim=2) #[b, neib, hid, hid]
        # print(attention_scores.shape)

        # Weighted sum of value vectors
        weighted_values = torch.matmul(attention_scores.transpose(2, 3), V_h) # [b, neib, hid, hid] * [b, neib, hid, head]
        output = weighted_values.squeeze(1).sum(dim = 1)  # get weighted sum # [b, hid, hid]
        # print(output.shape)

        return output.mean(dim=2) # take mean of all attention heads # [b, hid]

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
        self.num_neighbor = num_neighbor
        
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
        
    def forward(self, center_embedding, neighbor_embedding):
        h_in1 = center_embedding # for first residual connection
        attn_out = self.attention(center_embedding, neighbor_embedding)
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
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
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
#======= graphtransformer layer test ============
# Model parameters
# in_dim = 768  # Input feature dimension
# out_dim = 768  # Output feature dimension
# num_heads = 8
# num_neighbors = 30  # Assume each node has 30 neighbors

# # Initialize the Graph Transformer Layer
# graph_transformer = GraphTransformerLayer(
#     in_dim=in_dim,
#     out_dim=out_dim,
#     num_heads=num_heads,
#     num_neighbor=num_neighbors,
#     dropout=0.1,
#     layer_norm=True,
#     batch_norm=True,
#     residual=True,
#     use_bias=True
# )

# # Generate synthetic input data
# batch_size = 10
# center_embedding = torch.randn(batch_size, in_dim)  # Feature vector for the center node
# neighbor_embeddings = torch.randn(batch_size, num_neighbors, in_dim)  # Feature vectors for the neighbors

# # Perform a forward pass through the model
# output = graph_transformer(center_embedding, neighbor_embeddings)

# # Print the output to verify results
# print("Output Shape of gt layer:", output.shape)
# # print("Output Tensor:", output)

#--------
# # Assuming the attention layer and dimensions are already defined
# in_dim = 768  # Ensure this is set correctly according to your model's expected input dimension
# out_dim = 768  # Output dimension set in your MultiHeadAttentionLayer initialization
# num_heads = 8  # Number of attention heads
# num_neighbors = 30  # Each node has 30 neighbors

# # Initialize the attention layer
# attention_layer = MultiHeadAttentionLayer(in_dim, out_dim, num_heads, num_neighbors, use_bias=True)

# # Prepare batch data
# batch_size = 10  # Define a batch size for testing
# center_embeddings = torch.rand(batch_size, in_dim)  # Batch of center node embeddings
# neighbor_embeddings = torch.rand(batch_size, num_neighbors, in_dim)  # Corresponding neighbors

# # Get output from the attention layer
# output = attention_layer(center_embeddings, neighbor_embeddings)
# print(output.shape) 

# ======= multihead attention test ============
# # Example usage:
# in_dim = 768  # Dimension of input embeddings
# out_dim = 768  # Dimension of output embeddings
# num_heads = 8  # Number of attention heads
# use_bias = True  # Whether to use bias in linear transformations
# num_neighbor = 30

# # Create the attention layer
# attention_layer = MultiHeadAttentionLayer(in_dim, out_dim, num_heads, num_neighbor, use_bias)

# embs = []
# for _ in range(30):
#     # Example data: center embedding and 30 neighbor embeddings
#     center_embedding = torch.rand(1, in_dim)  # One center node embedding
#     neighbor_embeddings = torch.rand(30, in_dim)  # 30 neighbor node embeddings
#     embs.append([center_embedding, neighbor_embeddings])

# # Get output from the attention layer
# output = attention_layer(center_embedding, neighbor_embeddings)
# print(output.shape)  # Expected shape: (out_dim,)
