import torch
from dl import create_dataloader
from adaglm import AdaGLM
from transformers import BertTokenizer
from tqdm import tqdm

def load_model(model_path, device, net_params):
    model = AdaGLM(net_params).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def inference(model, data_loader, device, filename):
    model.eval()  # Set the model to evaluation mode
    all_embeddings = []
    all_indices = []
    batch_count = 0
    num_batches = 4
    with torch.no_grad():
        for batch in tqdm(data_loader):
            node_index = batch['node_index']
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            center_lap_encodings = batch['lap_embedding'].to(device)
            neighbor_lm_embeddings = batch['neighbor_lm_embedding'].to(device)
            neighbor_lap_encodings = batch['neighbor_lap_embedding'].to(device)

            one_hop_embedding = batch['one_hop_embedding'].to(device)
            one_hop_mask = batch['one_hop_mask'].to(device)
            
            outputs = model(input_ids, token_type_ids, attention_mask, neighbor_lm_embeddings, center_lap_encodings, neighbor_lap_encodings, one_hop_embedding, one_hop_mask)
            gtf_embeddings, _, _, _ = outputs  # Assuming outputs are embeddings

            gtf_embeddings = gtf_embeddings.cpu()
            # node_index = node_index.cpu()

            all_embeddings.append(gtf_embeddings)
            all_indices.extend(node_index)
            # batch_count += 1
            # if batch_count >= num_batches:
            #     break
    
    # Concatenate all embeddings and convert indices to a tensor
    all_embeddings = torch.cat(all_embeddings, dim=0)
    # all_indices = torch.tensor(all_indices, dtype=torch.long)
    print(all_embeddings.shape)

    # Save both tensors in a dictionary as a single .pt file
    torch.save({'indices': all_indices, 'embeddings': all_embeddings}, filename)
   

    return all_indices, all_embeddings


# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/gpfsnyu/scratch/qz2086/AdaGLM/checkpoints/unmasked/avglm_6_6_bottleneck_apt_wgtf_12_10_w0_90.pth'

net_params = {
    'lm_dim': 768,  # Assuming the output dimension of the language model
    'hidden_dim': 768,
    'out_dim': 768,
    'n_heads': 12,
    'in_feat_dropout': 0,
    'dropout': 0,
    'L': 12,  # Number of layers
    'readout': 'mean',
    'layer_norm': True,
    'batch_norm': True,
    'residual': True,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'lap_pos_enc': True,
    'pos_enc_dim': 768,  # Dimension of the Laplacian positional encoding
    'num_neighbors': 10,
    'gtf_neib_num': 6,
    'node_pos': 6,
    'lm_model': 'bert-base-uncased',
    'mode_loss': 0,
    'tau': 0.5,
    'if_adaptor': True,
    'if_gtf': True,
    'lm':'unmasked', #'unmasked'
    'mlm_prob': 0.15,
    'mode': 'inference', # 'inference',
    'state':'bottleneck'
}
print(net_params)
model = load_model(model_path, device, net_params)

data_dir = '/gpfsnyu/scratch/qz2086/AdaGLM/data'
domain = 'combined'
tensor_file_path = f'{data_dir}/{domain}/e_g/tensor_data_10_shuffled.pt'
text_file_path = f'{data_dir}/{domain}/e_g/e_g_tokenized_text_abs_full.pt'


batch_size = 64
dataloader = create_dataloader(tensor_file_path, text_file_path, batch_size, num_workers = 4, is_distributed = False)

filename = f'{data_dir}/{domain}/avglm_6_6_bottleneck_apt_wgtf_12_10_w0_90.pt'
print(filename)
# Perform inference
indicies, emb = inference(model, dataloader, device, filename)
# print(indicies.shape)
# print(emb.shape)


# # Create dataloader
# tensor_file_path = '/path/to/tensor_data.pt'
# text_file_path = '/path/to/node.txt'
# dataloader = create_dataloader(tensor_file_path, text_file_path, batch_size, num_workers = 30, is_distributed = False)

