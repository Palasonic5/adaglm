import torch
import torch.nn as nn
import torch.nn.functional as F

from adaglm import AdaGLM


# Parameters for AdaGLM
net_params = {
    'lm_dim': 768,  # Assuming the output dimension of the language model
    'hidden_dim': 128,
    'out_dim': 728,
    'n_classes': 3,
    'n_heads': 8,
    'in_feat_dropout': 0.1,
    'dropout': 0.2,
    'L': 10,  # Number of layers
    'readout': 'mean',
    'layer_norm': True,
    'batch_norm': True,
    'residual': True,
    'device': torch.device('cpu'),
    'lap_pos_enc': True,
    'pos_enc_dim': 768,  # Dimension of the Laplacian positional encoding
    'num_neighbors': 30
}

model = AdaGLM(net_params)
model = model.to(net_params['device'])

# Simulated data
center_text = "Example center node text"
neighbor_feature = torch.randn(30, 768)  # 30 neighbors with features
center_lap_embedding = torch.randn(768)  # Random positional encoding
neighbor_lap_embedding = torch.randn(30, 768)  # Pos encodings for neighbors
neg_embedding = torch.randn(30, 768)
# As the model expects the embeddings to be already processed by a language model,
# we simulate these embeddings
center_embedding_simulated = torch.randn(1, net_params['lm_dim'])  # Simulated output from the LM

# Forward pass through the model
output = model(
    center_text,  # Passing simulated LM output
    neighbor_feature,
    center_lap_embedding,
    neighbor_lap_embedding
)

# Print the output to see what it looks like
# print("Output shape:", output.shape)
# print("Output tensor:", output)


