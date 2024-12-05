
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F

# from language_model import unmasked_bert

class AdaptorLayer(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state
        if self.state == 'original':
        # self.emb_size = emb_size
            self.layers = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
                # nn.ReLU(),
                # nn.Linear(hid_size, 1),
            )
        elif self.state == 'lmdimension':
            self.weights = nn.Parameter(torch.randn(13,768))
        elif self.state == 'bottleneck':
            self.layers = nn.Sequential(
                nn.Linear(13 * 768, 768),
                nn.ReLU(),
                nn.Linear(768, 1),
                nn.ReLU(),
                nn.Linear(1, 768),
            )
        else:
            raise ValueError('unsupported adaptor type')
        

    def forward(self, lm_embedding):

        preds = lm_embedding
        if self.state == 'bottleneck':
            # preds = torch.stack(preds, dim = 1)
            input_tensor = preds.view(preds.size(0), -1)
            # print(input_tensor.shape, flush = True)
            out = self.layers(input_tensor)
            # print(out.shape)
            return out

        if self.state == 'lmdimension':
            # preds = lm_embedding.permute(0, 2, 1)
            weighted = preds * self.weights
            output = weighted.sum(dim = 1)
            print(output.shape)
            return output
            
        # adaptive layer: stack - project - softmax - get retain - score-weighted feature
        # pps = torch.stack([torch.tensor(p) for p in preds], dim=0)   #[batch, layer, feat]
        retain_score = self.layers(preds) # [batch, layer, 1]
        retain_score = retain_score.squeeze(-1) #[batch, layer]
        retain_score = torch.softmax(retain_score, dim = 1) # [batch, layer]
        retain_score = retain_score.unsqueeze(1)  #[batch, 1, layer]

        out = torch.matmul(retain_score, preds).squeeze()   #[batch, 1, layer] * [layer, feat] = [batch, 1, feat]
        # print(out.shape)
        # out = out.permute(0,2,1)
        print(out.shape, flush = True)
        return out

        








# adaptor = AdaptorLayer(unmasked_bert)

# # text = "This is a sample text to test the adaptor model."
# texts = ["Hello, world!", "Testing BERT model.", "Another sentence to increase batch size.", 'eoifuo osuf siduf']

# output = adaptor(texts)
# # print("Output from Adaptor:", output)
# print('output size:', output.shape)

