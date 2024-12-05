from transformers import BertForMaskedLM
import torch
import torch.nn as nn


class DecoderModel(nn.Module):
    def __init__(self, config):
        super(DecoderModel, self).__init__()
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        # Isolate the decoder layer
        self.decoder = model.cls.predictions.decoder
        self.config = config
    
    def forward(self, hidden_states):
        # hidden_states expected shape: [batch_size, sequence_length, hidden_size]
        logits = self.decoder(hidden_states)
        return logits

# # Configuring the custom model
# config = model.config
# decoder_model = DecoderModel(decoder, config).to('cuda')
