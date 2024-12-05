import torch
from torch.utils.data import Dataset, DataLoader
import json

class AdaGLMDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, labels = False):
        """
        Args:
            jsonl_file (string): Path to the jsonl file with annotations.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process the text data.
        """
        self.data = []
        self.labels = labels
        with open(jsonl_file, 'r') as file:
            for line in file:
                entry = json.loads(line)
                self.data.append(entry)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        # Process the text through the tokenizer
        center_text = self.tokenizer(entry['center text'], padding='max_length', truncation=True, max_length=512, return_tensors="pt").input_ids.squeeze(0)
        # Convert lists to tensors
        neighbor_emb = torch.tensor(entry['neighbor embedding'])
        neighbor_lap_emb = torch.tensor(entry['neighbor lap emb'])
        negative_samples = torch.tensor(entry['negative samples'])
        if self.labels:
            label = torch.tensor(entry['label'])  
            return center_text, neighbor_emb, neighbor_lap_emb, negative_samples, label

        return center_text, neighbor_emb, neighbor_lap_emb, negative_samples


# # Example usage of AdaGLMDataset
# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# dataset = AdaGLMDataset('input.jsonl', tokenizer)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
