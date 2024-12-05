import torch
from torch import nn
from adaglm import AdaGLM
from dl import NodeDataDataset, create_dataloader

import torch
import numpy as np
from transformers import BertTokenizer

def generate_synthetic_data(batch_size, in_dim, num_neighbors, num_negatives):
    # Initialize a tokenizer (assuming using BERT as in the rest of the AdaGLM setup)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Generate some sample texts
    sample_texts = ["This is a sample sentence.", "Example of a short text.", "Here is another one."]
    center_texts = [np.random.choice(sample_texts) for _ in range(batch_size)]
    # encoded_texts = tokenizer(center_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").input_ids

    # Random embeddings for neighbors
    neighbor_features = torch.randn(batch_size, num_neighbors, in_dim)
    # Random embeddings for negative samples
    negative_samples = torch.randn(batch_size, num_negatives, in_dim)
    # Random Laplacian positional encodings for centers and neighbors
    center_lap_encodings = torch.randn(batch_size, in_dim)
    neighbor_lap_encodings = torch.randn(batch_size, num_neighbors, in_dim)
    
    return center_texts, neighbor_features, negative_samples, center_lap_encodings, neighbor_lap_encodings

# Now the center_texts are actual strings tokenized properly for BERT.


# Model parameters
net_params = {
    'lm_dim': 768,  # Assuming the output dimension of the language model
    'hidden_dim': 768,
    'out_dim': 768,
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
    'num_neighbors': 30,
    'gtf_neib_num': 20,
    'node_pos': 10
}

# Create the AdaGLM model
model = AdaGLM(net_params)
model.to('cpu')  # Assuming testing on CPU; change to 'cuda' if using GPU

# Generate test data
# batch_size = 10  # Number of data points in the batch
# center_texts, neighbor_features, negative_samples, center_lap_encodings, neighbor_lap_encodings = generate_synthetic_data(
#     batch_size, net_params['lm_dim'], net_params['num_neighbors'], net_params['num_neighbors'])
data_dir = '/scratch/qz2086/AdaGLM/data'
domain = 'Economics'

tensor_file_path = f'{data_dir}/{domain}/tensor_data_full.pt'
text_file_path = f'{data_dir}/{domain}/node.txt'
batch_size = 32

dataloader = create_dataloader(tensor_file_path, text_file_path, batch_size)
for batch in dataloader:
    # center_texts, center_lap_encodings, neighbor_features, neighbor_lap_encodings = batch
    center_texts = batch['text']
    # center_lap_encodings = batch['lap_embedding']
    neighbor_lm_embeddings = batch['neighbor_lm_embedding']
    print(neighbor_lm_embeddings.shape)
    # neighbor_lap_encodings = batch['neighbor_lap_embedding']
    # print(neighbor_lm_embedding.shape)

    # Run the model forward pass
    outputs = model(center_texts, neighbor_lm_embeddings)
    gtf_embedding, center_lin, neib_loss = outputs

    # Calculate a hypothetical loss (assuming some functionality for it)
    loss = model.loss(outputs, tau=0.1)
    print("Loss:", loss.item())
    print("Output shapes:", gtf_embedding.shape)


# def generate_test_data(batch_size, feature_dim, num_neighbors, num_negatives):
#     # Generate synthetic data for testing
#     out = []
#     for _ in batch_size:
#         center_text = ["sample text"]  # Simulate batch of text inputs
#         neighbor_feature = torch.randn(num_neighbors, feature_dim)  # Neighbor features
#         center_lap_embedding = torch.randn(feature_dim)  # Laplacian embedding for center
#         neighbor_lap_embedding = torch.randn(num_neighbors, feature_dim)  # Laplacian embedding for neighbors
#         negative_embedding = torch.randn(num_neighbors, feature_dim)
#         example = [center_text, neighbor_feature, center_lap_embedding, neighbor_lap_embedding]
#         out.append(example)
#     return out

# def test_model_forward_and_loss():
#     net_params = {
#     'lm_dim': 768,  # Assuming the output dimension of the language model
#     'hidden_dim': 128,
#     'out_dim': 728,
#     'n_classes': 3,
#     'n_heads': 8,
#     'in_feat_dropout': 0.1,
#     'dropout': 0.2,
#     'L': 10,  # Number of layers
#     'readout': 'mean',
#     'layer_norm': True,
#     'batch_norm': True,
#     'residual': True,
#     'device': torch.device('cpu'),
#     'lap_pos_enc': True,
#     'pos_enc_dim': 768,  # Dimension of the Laplacian positional encoding
#     'num_neighbors': 30
# }
#     model = AdaGLM(net_params)
#     model.to(net_params['device'])
    
#     # Generate synthetic data
#     center_text, neighbor_feature, center_lap_embedding, neighbor_lap_embedding = generate_test_data(
#         batch_size=10, 
#         feature_dim=net_params['hidden_dim'], 
#         num_neighbors=net_params['num_neighbors']
#     )
    
#     # Run the forward pass of the model
#     adaptor_embedding, final_output = model(center_text, neighbor_feature, center_lap_embedding, neighbor_lap_embedding)
    
#     # Now adaptor_embedding and final_output are directly from the model
#     # Continue to use them in your loss function or further processing
    
#     print("Adaptor Embedding Shape:", adaptor_embedding.shape)
#     print("Final Output Shape:", final_output.shape)

# # Run the test function
# test_model_forward_and_loss()
