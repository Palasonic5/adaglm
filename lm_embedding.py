from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm


def get_unmasked_embedding(texts, model, tokenizer):
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    # encoded_input = encoded_input.to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
    hidden_states = outputs.hidden_states
    cls_embeddings = hidden_states[-1][:, 0, :] 
    return cls_embeddings

def process_file(input_file, output_file, model, tokenizer):
    embeddings = []
    node_indices = []
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    i = 0
    c = 5
    with open(input_file, 'r') as f:
        progress_bar = tqdm(f, total=total_lines, desc="Processing")
        for line in progress_bar:
            node_index, text = line.strip().split(maxsplit=1)
            embedding = get_unmasked_embedding(text, model, tokenizer)
            embeddings.append(embedding)
            node_indices.append(node_index)

            
            # Save each embedding with its node index
            # output_file = f"{output_dir}/{node_index}_embedding.pt"
        paper_to_idx = {paper_id: idx for idx, paper_id in enumerate(node_indices)}
        torch.save({
            'node_index': paper_to_idx,
            'embedding': embeddings
        }, output_file)

# Load model and tokenizer
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name, output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Define directories and domain
data_dir = '/gpfsnyu/scratch/qz2086/AdaGLM/data'
# output_dir = '/gpfsnyu/scratch/qz2086/AdaGLM/data'
domain = 'Mathematics'

print('preparing lm emebdding for', domain)

# Define the input and output file paths
input_file = f'{data_dir}/{domain}/node_text_abs_full.txt'
output_file = f'{data_dir}/{domain}/lm_embeddings_abs.pt'  # Note the .pt extension for PyTorch tensor files
print(input_file)

# Process the file and generate embeddings
process_file(input_file, output_file, model, tokenizer)

print('file saved')



# with open(input_file, 'r') as f, open(output_file, 'w') as fout:
#     for line in f:
#         node_index, text = line.strip().split(maxsplit=1)
#         embedding = get_unmasked_embedding(text, model_name)
        
#         # Convert embedding tensor to list and write to output file
#         embedding_list = embedding.squeeze().tolist()  # Squeeze to remove batch dimension
#         embedding_str = ' '.join(map(str, embedding_list))
#         fout.write(f"{node_index} {embedding_str}\n")





# class UnmaskedBERT:
#     def __init__(self, model_name='bert-base-uncased'):
#         # Initialize and load the tokenizer and model at the instance creation
#         # self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
#         self.model.eval()  # Set model to evaluation mode
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)  # Move model to the appropriate device

#     def get_cls_embeddings(self, input_ids, token_type_ids, attention_mask):
#         # Encode the input texts
#         # encoded_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
#         # encoded_input = encoded_input.to(self.device)  # Move encoded input to the same device as the model

#         # Perform a forward pass to get the hidden states
#         with torch.no_grad():
#             outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
#         # Extract hidden states
#         hidden_states = outputs.hidden_states

#         # Extract the CLS token's embeddings from each layer
#         cls_embeddings = torch.stack([layer[:, 0, :] for layer in hidden_states], dim=1)

#         return cls_embeddings  # [batch, num_layer, feat_dim]



            # torch.save(embedding, output_file)

            # node_index, text = line.strip().split(maxsplit=1)
            # # print(node_index)
            # # print(text)
            # embedding = get_unmasked_embedding(text, model, tokenizer)
            # embeddings.append(embedding)
            # node_indices.append(int(node_index))
            # if i > c:
            #     break
            # i += 1
    
    # Stack all embeddings into a single tensor for efficient storage
    # embeddings_tensor = torch.cat(embeddings, dim=0)
    # node_indices_tensor = torch.tensor(node_indices, dtype=torch.long)
    
    # Save both embeddings and indices in a single file
    # torch.save({
    #     "node_indices": node_indices_tensor,
    #     "embeddings": embeddings_tensor
    # }, output_file)
