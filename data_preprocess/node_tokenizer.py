from transformers import BertTokenizer
import torch
from tqdm import tqdm

class NodeDataDataset:
    def __init__(self, text_file_path, processed_text_file_path):
        # Load tensor data
        
        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load and tokenize text data
        node_texts = {}

        with open(text_file_path, 'r') as file:
            for line in tqdm(file):
                parts = line.strip().split(maxsplit=1)
                node_index = int(parts[0])
                text = parts[1]
                tokenized_text = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512) # make sure sequence length is the same
                node_texts[node_index] = tokenized_text

        self.text_data = node_texts
        
        # Save tokenized text data directly to a .pt file
        torch.save(node_texts, processed_text_file_path)


data_dir = '/gpfsnyu/scratch/qz2086/AdaGLM/data'
domain = 'Geology'
processed_text_file_path = f'{data_dir}/{domain}/tokenized_text_abs_full.pt'
text_file_path = f'{data_dir}/{domain}/node_text_abs_full.txt'

print('preparing tokenized text for:', domain)

data = NodeDataDataset(text_file_path, processed_text_file_path)
print('finished tokenizing texts')