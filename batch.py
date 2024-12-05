import json
from tqdm import tqdm
import torch

print('preparing batch for econ', flush = True)

def load_embeddings(file_path):
    print('loading embeddings')
    embeddings = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            node_index = int(parts[0])
            embeddings[node_index] = list(map(float, parts[1].split()))
    return embeddings

def load_neighbors(file_path):
    neighbors = {}
    with open(file_path, 'r') as file:
        neighbors = json.load(file)  # Assumes the entire file is a single JSON object
    # Convert string indices to integers
    return neighbors

def load_text(file_path):
    print('loading texts')
    node_texts = {}
    with open(file_path, 'r') as file:
        for line in tqdm(file):
            parts = line.strip().split(maxsplit=1)
            node_index = int(parts[0])
            node_texts[node_index] = parts[1]
    return node_texts

def sampler(node_data, k):
    paper_idx = node_data['paper_index']
    one_hop = node_data['1-hop']
    two_hop = node_data['2-hop']
    if len(one_hop) >= k:
        one_hop = one_hop[:k]
        neighbor = one_hop
    else:
        # num_one_hop = len(one_hop)
        # neighbor.extend(two_hop)
        if len(one_hop) + len(two_hop) >= k:
            num = k -len(one_hop)
            two_hop = two_hop[:num]
            # one_hop.extend(two_hop)
            neighbor = one_hop + two_hop
        else:
            # print('does not have enough neighbor for node:', paper_idx)
            return None, None
    return one_hop, neighbor

def prepare_and_save_data(lm_idx, lm_emb, lap, neighbors, tensor_file_path):
    tensor_data = []
    i = 0
    c = 50
    lap_idx = lap['paper_ids']
    lap_paper_to_idx = {str(paper_id.item()): idx for idx, paper_id in enumerate(lap_idx)}
    lap_emb = lap['positional_encodings']

    isolated_count = 0

    for node_index in tqdm(lm_idx):
        node_index = str(node_index)

        if node_index in neighbors:

            lap_idx_tmp = lap_paper_to_idx[node_index]
            lap_pe = lap_emb[lap_idx_tmp]
            neighbor_info = neighbors[node_index]
            one_hop_idx, neighbor_indices = sampler(neighbor_info, 10)

            if neighbor_indices:
                neib_lm_idx = [lm_idx[b] for b in neighbor_indices]
                neighbor_lm_embeddings = [lm_emb[n] for n in neib_lm_idx]
                neighbor_lm_embeddings = torch.stack(neighbor_lm_embeddings).squeeze()

                one_hop_lm_idx = [lm_idx[b] for b in one_hop_idx]
                one_hop_lm_embeddings = [lm_emb[n] for n in one_hop_lm_idx]
                one_hop_lm_embeddings = torch.stack(one_hop_lm_embeddings).squeeze()


                neib_lap_idx = [lap_paper_to_idx[nb] for nb in neighbor_indices]
                neib_pe = [lap_emb[n] for n in neib_lap_idx]
                neib_pe = torch.stack(neib_pe).squeeze()
                
                
                # neighbor_lap_embeddings = [lap_embeddings.get(neighbor_idx, [0]*768) for neighbor_idx in neighbor_indices]
                tensor_entry = {
                    'node_index': node_index,  # Preserve the node index for matching
                    'lap_embedding': lap_pe,
                    'one_hop': one_hop_lm_embeddings,
                    'neighbor_lm_embedding': neighbor_lm_embeddings,
                    'neighbor_lap_embedding': neib_pe
                }
                tensor_data.append(tensor_entry)
            else:
                isolated_count += 1

    torch.save(tensor_data, tensor_file_path)
    print('isolated node count:', isolated_count)
    return tensor_data



data_dir = '/gpfsnyu/scratch/qz2086/AdaGLM/data'
sub_dataset = 'Mathematics'

print('preparing batch file for:', sub_dataset)
# # Load data from files
print('loading lm embeddings', flush = True)
# lm_embeddings = load_embeddings(f'{data_dir}/{sub_dataset}/lm_embeddings.txt')
lm_embeddings = torch.load(f'{data_dir}/{sub_dataset}/lm_embeddings_abs.pt')
lm_idx = lm_embeddings['node_index']
# print(lm_idx)
lm_emb = lm_embeddings['embedding']
# print('loading lap embeddings')
print('loading laplacian embedding', flush = True)
lap_embedding = torch.load(f'{data_dir}/{sub_dataset}/laplacian_encodings_math.pt')

print('loading neighbor file', flush = True)
neighbors = load_neighbors(f'{data_dir}/{sub_dataset}/pr_neighbors.json')

prepare_and_save_data(lm_idx, lm_emb, lap_embedding, neighbors, f'{data_dir}/{sub_dataset}/tensor_data_10_neighbor_812.pt')

print('data saving finished')