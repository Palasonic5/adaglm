import torch
from data_loader import create_dataloader_index
from adaglm import AdaGLM
from transformers import BertTokenizer
from tqdm import tqdm

# import torch
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path, device, net_params):
    model = AdaGLM(net_params).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def plot_heatmap(retain_score, file_path):
    # Convert the retainment scores to numpy for visualization
    scores = retain_score.cpu().numpy()

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 2))  # Adjust the figure size as needed
    sns.heatmap(scores.reshape(1, -1), annot=True, cmap="viridis", cbar=True, xticklabels=False)
    
    # plt.title(f"Retainment Scores for Input {batch_index}")
    plt.xlabel("Layer")
    plt.ylabel("Retainment Score")
    plt.show()
    plt.savefig(file_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()

def inference(model, data, device, filename):
    model.eval()  # Set the model to evaluation mode
    all_embeddings = []
    all_indices = []
    batch_count = 0
    num_batches = 4
    with torch.no_grad():
        # batch = next(iter(data_loader))
        batch = data
        node_index = batch['node_index']
        input_ids = batch['input_ids'].unsqueeze(0).to(device)
        print(input_ids.shape, flush = True)
        token_type_ids = batch['token_type_ids'].unsqueeze(0).to(device)
        attention_mask = batch['attention_mask'].unsqueeze(0).to(device)

        center_lap_encodings = batch['lap_embedding'].unsqueeze(0).to(device)
        print(center_lap_encodings, flush = True)
        neighbor_lm_embeddings = batch['neighbor_lm_embedding'].unsqueeze(0).to(device)
        print(neighbor_lm_embeddings, flush = True)
        neighbor_lap_encodings = batch['neighbor_lap_embedding'].unsqueeze(0).to(device)
        print(neighbor_lap_encodings, flush = True)

        one_hop_embedding = batch['one_hop_embedding'].unsqueeze(0).to(device)
        print(one_hop_embedding, flush = True)
        one_hop_mask = batch['one_hop_mask'].unsqueeze(0).to(device)
        print(one_hop_mask, flush = True)
        
        outputs = model(input_ids, token_type_ids, attention_mask, neighbor_lm_embeddings, center_lap_encodings, neighbor_lap_encodings, one_hop_embedding, one_hop_mask)
        gtf_embeddings, center_lin, neib_loss, lm_loss = outputs  # Assuming outputs are embeddings

        lm = model.language_model(input_ids, token_type_ids, attention_mask)

        # adaptor output
        # preds = center_lin
        print(lm, flush = True)
        retain_score = model.adaptor.layers(lm)  # [batch, layer, 1]
        print(retain_score.shape, flush = True)
        retain_score = retain_score.squeeze(-1)  # [batch, layer]
        print(retain_score.shape, flush = True)
        retain_score = torch.softmax(retain_score, dim=1)
        print(retain_score.shape, flush = True)
        print(retain_score)

        plot_heatmap(retain_score, 'retain_score_e_g_infwithlm_512.png')

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
    # torch.save({'indices': all_indices, 'embeddings': all_embeddings}, filename)
   

    return all_indices, all_embeddings


# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/gpfsnyu/scratch/qz2086/AdaGLM/checkpoints/masked/inf_lmloss_e_g_checkpoint_6_6_noapt_wgtf_12_10_w0_40.pth'

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
    'lm':'masked', #'unmasked'
    'mlm_prob': 0.15,
    'mode': 'inference' # 'inference'
}
print(net_params)
model = load_model(model_path, device, net_params)

data_dir = '/gpfsnyu/scratch/qz2086/AdaGLM/data'
domain = 'combined'
tensor_file_path = f'{data_dir}/{domain}/e_g/tensor_data_10_shuffled.pt'
text_file_path = f'{data_dir}/{domain}/e_g/e_g_tokenized_text_abs_full.pt'
# selected_idx_path = {10: [2332735496, 1601632087, 1597139360], 25:  [3089153405, 2297639561, 1513804007]
# 50:  [3124697985, 2966179627, 3023293734]
# 100:  [2018014132, 2058982498, 2100458410]
# 150:  [1578153808, 2018090882, 3122674340]
# 200:  [2062346067, 2139853108, 1979370126]
# 250:  [2009867146, 1967247616, 2083028955]
# 300:  [2157936678, 2026457720, 2057109618]
# 350:  [3121326953, 2087450332, 1980320150]
# 400:  [2114089828, 2067087527, 2014035003]
# 450:  [2335720368, 2035722485, 2018933200]
# 500:  [1978864333, 2259193740, 1571082978]
# 512:  [1992956121, 2027232499, 3121173810]}


# batch_size = 1
dataloader = create_dataloader_index(tensor_file_path, text_file_path, index = 2027232499, batch_size = 1, num_workers = 4, is_distributed = False)

filename = f'{data_dir}/{domain}/inf_lm_6_6_w2apt_wgtf_15_10_w0_230.pt'
print(filename)
# Perform inference
indicies, emb = inference(model, dataloader, device, filename)
# print(indicies.shape)
# print(emb.shape)


# # Create dataloader
# tensor_file_path = '/path/to/tensor_data.pt'
# text_file_path = '/path/to/node.txt'
# dataloader = create_dataloader(tensor_file_path, text_file_path, batch_size, num_workers = 30, is_distributed = False)

