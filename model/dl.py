import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence


class NodeDataDataset(Dataset):
    def __init__(self, tensor_file_path, text_file_path):
        # Load tensor data
        self.tensor_data = torch.load(tensor_file_path)
        self.text_data = torch.load(text_file_path)
        # self.lap_data = torch.load(lap_file)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def __len__(self):
        # assert (len(self.tensor_data)) == len(self.text_data)
        return len(self.tensor_data)

    # def sample_neighbors(self, all_neighbors, num):
    def pad_one_hop_embeddings_with_mask(self, embeddings, max_length=10, padding_value=0):
        # print('before masking', embeddings.shape)
        if embeddings.dim() == 1:  # If the embeddings are 1D
            embeddings = embeddings.unsqueeze(0)
        current_length = embeddings.shape[0]
    
        # Create a mask for the actual data entries
        mask = torch.ones(current_length, dtype=torch.bool)
        
        # Calculate how much padding is needed (if any)
        if current_length < max_length:
            padding_needed = max_length - current_length
            
            padding_tensor = torch.full((padding_needed, embeddings.shape[1]), padding_value)
            
            mask = torch.cat([mask, torch.zeros(padding_needed, dtype=torch.bool)], dim=0)
            
            padded_embeddings = torch.cat([embeddings, padding_tensor], dim=0)
        else:
            padded_embeddings = embeddings
        
        return padded_embeddings, mask

    def __getitem__(self, idx):
        # Fetch the tensor data entry
        tensor_entry = self.tensor_data[idx]
        node_index = tensor_entry['node_index']

        # Fetch the corresponding text data using node index
        text = self.text_data[int(node_index)]
        # encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = text['input_ids'].squeeze(0)
        token_type_ids = text['token_type_ids'].squeeze(0)
        attention_mask = text['attention_mask'].squeeze(0)
        center_pe = tensor_entry['lap_embedding']
        neib_emb = tensor_entry['neighbor_lm_embedding']
        one_hop_embedding = tensor_entry['one_hop'] 
        one_hop_embedding_padded, mask = self.pad_one_hop_embeddings_with_mask(one_hop_embedding)
        neib_pe = tensor_entry['neighbor_lap_embedding']
        # lap = self.lap_data[node_index]

        # Prepare the data dictionary
        data = {
            'node_index': node_index,
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'lap_embedding': center_pe,
            'neighbor_lm_embedding': neib_emb,
            'neighbor_lap_embedding': neib_pe,
            'one_hop_embedding': one_hop_embedding_padded,
            'one_hop_mask': mask.float()
        }
        return data


class ChunkNodeDataDataset(Dataset):
    def __init__(self, tensor_file_paths, text_file_path, chunk_size=10000):
        # Load the text data at once
        self.text_data = torch.load(text_file_path)
        
        # Setup for chunk-wise loading of tensor data
        # self.tensor_file_path = tensor_file_path
        # self.chunk_size = chunk_size
        # self.current_chunk = 0
        # self.current_data = []
        # self.total_length = self.get_total_entries()
        self.tensor_file_paths = tensor_file_paths
        self.current_chunk_index = -1  # No chunk is loaded initially
        self.current_data = None
        
        # Calculate total length by loading each chunk and getting its length
        self.total_length = 0
        self.chunk_lengths = []
        for path in tensor_file_paths:
            data = torch.load(path, map_location='cpu')

            chunk_length = len(data)

            self.chunk_lengths.append(chunk_length)
            self.total_length += chunk_length

    def pad_one_hop_embeddings_with_mask(self, embeddings, max_length=10, padding_value=0):
        # print('before masking', embeddings.shape)
        if embeddings.dim() == 1:  # If the embeddings are 1D
            embeddings = embeddings.unsqueeze(0)
        current_length = embeddings.shape[0]
    
        # Create a mask for the actual data entries
        mask = torch.ones(current_length, dtype=torch.bool)
        
        # Calculate how much padding is needed (if any)
        if current_length < max_length:
            padding_needed = max_length - current_length
            
            padding_tensor = torch.full((padding_needed, embeddings.shape[1]), padding_value)
            
            mask = torch.cat([mask, torch.zeros(padding_needed, dtype=torch.bool)], dim=0)
            
            padded_embeddings = torch.cat([embeddings, padding_tensor], dim=0)
        else:
            padded_embeddings = embeddings
        
        return padded_embeddings, mask
        
    def get_total_entries(self):
        # Load the first chunk to know total length
        chunk = torch.load(self.tensor_file_path, map_location='cpu', pickle_load=lambda storage, loc: storage)
        return len(chunk)

    def load_chunk(self, index):
        # Load a chunk of tensor data
        # self.current_data = torch.load(self.tensor_file_path, map_location='cpu', pickle_load=lambda storage, loc: storage[self.current_chunk * self.chunk_size:(self.current_chunk + 1) * self.chunk_size])
        # self.current_chunk += 1
        if self.current_chunk_index != index:
            self.current_data = torch.load(self.tensor_file_paths[index], map_location='cpu')
            self.current_chunk_index = index


    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Determine if current index is within the loaded chunk
        # local_idx = idx % self.chunk_size
        # chunk_idx = idx // self.chunk_size

        # # If idx requested is not in the current chunk, load the correct chunk
        # if chunk_idx != self.current_chunk:
        #     self.current_chunk = chunk_idx
        #     self.load_chunk()
        
        # # Fetch the tensor data entry
        local_idx = idx
        chunk_index = 0
        
        # Find the appropriate chunk index based on the local_idx
        for i, length in enumerate(self.chunk_lengths):
            if local_idx < length:
                chunk_index = i
                break
            local_idx -= length
        
        # Load the appropriate chunk
        self.load_chunk(chunk_index)
        
        # Fetch the tensor data entry
        tensor_entry = self.current_data[local_idx]
    
        # Fetch the corresponding text data using node index
        node_index = tensor_entry['node_index']
        text = self.text_data[int(node_index)]

        input_ids = text['input_ids'].squeeze(0)
        token_type_ids = text['token_type_ids'].squeeze(0)
        attention_mask = text['attention_mask'].squeeze(0)
        center_pe = tensor_entry['lap_embedding']
        neib_emb = tensor_entry['neighbor_lm_embedding']
        one_hop_embedding = tensor_entry['one_hop'] 
        one_hop_embedding_padded, mask = self.pad_one_hop_embeddings_with_mask(one_hop_embedding)
        neib_pe = tensor_entry['neighbor_lap_embedding']

        # Prepare the data dictionary
        data = {
            'node_index': node_index,
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'lap_embedding': center_pe,
            'neighbor_lm_embedding': neib_emb,
            'neighbor_lap_embedding': neib_pe,
            'one_hop_embedding': one_hop_embedding_padded,
            'one_hop_mask': mask.float()
        }
        
        return data


def create_dataloader(tensor_file_path, text_file_path, batch_size=4, num_workers=4, is_distributed=False):
    # Create an instance of the dataset
    dataset = NodeDataDataset(tensor_file_path, text_file_path)
    
    # Create a sampler for distributed training
    sampler = DistributedSampler(dataset) if is_distributed else None

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=num_workers)
    return dataloader


# def test_dataloader(dataloader):
#     print("Testing dataloader with up to 3 batches...")
#     for i, batch in enumerate(dataloader):
#         print(f"Batch {i + 1}")
#         print(f"Node Indexes: {batch['node_index']}")
#         print(f"Input IDs Shape: {batch['input_ids'].shape}")
#         print(f"Token Type IDs Shape: {batch['token_type_ids'].shape}")
#         print(f"Attention Mask Shape: {batch['attention_mask'].shape}")
#         print(f"Laplacian Embedding Shape: {batch['lap_embedding'].shape}")
#         print(f"Neighbor LM Embedding Shape: {batch['neighbor_lm_embedding'].shape}")
#         print(f"Neighbor Laplacian Embedding Shape: {batch['neighbor_lap_embedding'].shape}")
#         print(f"One-hop Embedding Shape: {batch['one_hop_embedding'].shape}")
#         print(f"One-hop Mask Shape: {batch['one_hop_mask'].shape}")
#         print("\n")
        
#         if i == 8:  # Stop after 3 batches
#             break

# # Example file paths -- replace these with actual file paths where your data is stored
# data_dir = '/gpfsnyu/scratch/qz2086/AdaGLM/data'
# domain = 'Economics'
# tensor_file_path = f'{data_dir}/{domain}/tensor_data_10_neighbor_812.pt'
# text_file_path = f'{data_dir}/{domain}/tokenized_text_abs_full.pt'
# batch_size = 32  # Adjust as needed
# num_workers = 0  # For testing, keep it simple with no parallel workers

# # Create a DataLoader
# dataloader = create_dataloader(tensor_file_path, text_file_path, batch_size, num_workers=num_workers, is_distributed=False)

# # Test the DataLoader
# test_dataloader(dataloader)