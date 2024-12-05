import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForMaskedLM, BertConfig
import random

class UnmaskedBERT:
    def __init__(self, model_name='bert-base-uncased', num_layers = 12, state = 'original'):
        
        self.model = BertModel.from_pretrained(model_name, num_hidden_layers = num_layers, output_hidden_states=True)
        self.model.eval()  # Set model to evaluation mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move model to the appropriate device
        self.state = state

    def get_cls_embeddings(self, input_ids, token_type_ids, attention_mask):
        
        with torch.no_grad():
            outputs = self.model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        
        hidden_states = outputs.hidden_states

        if self.state == 'lmdimension' or self.state == 'original':
            #cls_embeddings = torch.stack([layer[:, 0, :] for layer in hidden_states], dim=1)
            avg_embeddings = torch.stack([layer.mean(dim=1) for layer in hidden_states], dim=1)
            return avg_embeddings

        elif self.state == 'bottleneck':
            cls_embeddings = torch.stack([layer[:, 0, :] for layer in hidden_states], dim=1)
            # print('cls embedding shape:',cls_embeddings.shape)
            return cls_embeddings

        # return  avg_embeddings  #cls_embeddings  # [batch, num_layer, feat_dim]




class MaskedBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_layers = 12, mlm_prob = 0.15, mode = 'train'):
        super().__init__()
        self.config = BertConfig.from_pretrained(model_name, num_hidden_layers = num_layers, output_hidden_states = True)
        self.model = BertForMaskedLM.from_pretrained(model_name, num_hidden_layers = num_layers, output_hidden_states=True)
        # self.model = BertModel.from_pretrained(model_name, num_hidden_layers = num_layers, output_hidden_states=True)

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.mlm_prob = mlm_prob
        if mode == 'train':
            self.model.train()  # Set model to training mode
            # self.mlm_prob = mlm_prob
            print('whether training lm or not:', self.model.training, flush = True)
        elif mode == 'inference':
            # self.mlm_prob = 0
            self.model.eval()
        
    def mask_tokens(self, input_ids):
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        masking_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masking_indices] = -100
        input_ids[masking_indices] = self.tokenizer.mask_token_id

        return input_ids, labels

    def forward(self, input_ids, token_type_ids = None, attention_mask = None):

        input_ids, labels = self.mask_tokens(input_ids)
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        # outputs = self.model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        with torch.no_grad():
            outputs = self.model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, labels = labels, return_dict = True)

        hidden_states = outputs.hidden_states

        last_layer_embeddings = hidden_states[-1]

        cls_embeddings = torch.stack([layer[:, 0, :] for layer in hidden_states], dim=1)

        return cls_embeddings, last_layer_embeddings, labels

    def mlm_loss(self, embedding, labels):
        logits = self.model.cls(embedding).to(self.device)  # Assuming using the output layer from BertForMaskedLM
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # ignore_index=-100 ignores non-masked tokens
        return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))


        

# masked_bert = MaskedBERT()

# # Example data
# texts = ["Hello, how are you?", "See you later!"]
# encoded_inputs = masked_bert.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
# # print(encoded_inputs)
# input_ids, token_type_ids, attention_mask = encoded_inputs["input_ids"], encoded_inputs['token_type_ids'], encoded_inputs["attention_mask"]
# # print(input_ids.shape)
# # Move tensors to the same device as the model
# # input_ids, attention_mask = input_ids.to(masked_bert.device), attention_mask.to(masked_bert.device)

# # Compute loss and all layer embeddings
# cls_embeddings, last_layer_embeddings, labels = masked_bert(input_ids, token_type_ids, attention_mask=attention_mask)
# # print(f"Loss: {loss.item()}")
# mlm_loss = masked_bert.mlm_loss(last_layer_embeddings, labels)
# # print(loss)
# print(mlm_loss)
# print(f"clsembeddings:", cls_embeddings)
# print(cls_embeddings.shape)  # Example to access second layer embeddings
# print(last_layer_embeddings)
# print(last_layer_embeddings.shape)