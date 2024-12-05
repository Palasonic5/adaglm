import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptor import AdaptorLayer
from gt_layer_self_attention import GraphTransformerLayer
from transformers import BertModel, BertTokenizer
from language_model import UnmaskedBERT, masked_bert
from lm_decoder import DecoderModel


class MaskedAdaGLM(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        lm_dim = net_params['lm_dim'] # node_dim (feat is an integer), original name  'in_dim'
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        # n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        num_neighbors = net_params['num_neighbors']
        lm = net_params['lm']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        # self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.num_neighbors = net_params['num_neighbors']
        self.gtf_neib_num = net_params['gtf_neib_num']

        self.device = net_params['device']

        self.loss_neib_num = net_params['node_pos']

        # self.mod_pos_num = net_params['mod_pos']
        # self.mod_neg_num = net_params['mod_neg']

        if self.lm = 'unmasked':
            self.language_model = UnmaskedBERT(model_name = 'bert-base-uncased')
        else:
            self.language_model = masked_bert
            # self.mlm = decoder_model(layer = 1)
        
        # layers
        self.adaptor = AdaptorLayer(emb_size = 768, hid_size = 768)
        
        # self.embedding_h = nn.Embedding(lm_dim, hidden_dim) # node feat is an integer
        self.embedding_h = nn.Linear(lm_dim, hidden_dim)

        pos_enc_dim = net_params['pos_enc_dim']
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GraphTransformerLayer(in_dim=hidden_dim, 
                                                            out_dim=hidden_dim, 
                                                            num_heads=num_heads,
                                                            num_neighbor=self.gtf_neib_num, 
                                                            dropout=0.1,
                                                            layer_norm=True,
                                                            batch_norm=False,
                                                            residual=True,
                                                            use_bias=True) for _ in range(n_layers-1)])
        self.layers.append(GraphTransformerLayer(in_dim=hidden_dim, 
                                                out_dim=out_dim, 
                                                num_heads=num_heads,
                                                num_neighbor=self.gtf_neib_num, 
                                                dropout=0.1,
                                                layer_norm=True,
                                                batch_norm=False,
                                                residual=False,
                                                use_bias=True))


    def sample_neighbors(self, sequence_tensor, num_samples):
        b, seq, hid = sequence_tensor.shape
        # num_samples = self.gtf_neib_num

        if seq < num_samples:
            raise ValueError(f"Not enough embeddings to sample from. Required: {num_samples}, Available: {seq}")
        
        # Create a tensor of indices for uniform sampling without replacement
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        indices = torch.multinomial(torch.ones((b, seq)), num_samples, replacement=False).to(device)

        # Gather the sampled embeddings using the generated indices
        sampled_embeddings = torch.gather(sequence_tensor, 1, indices.unsqueeze(-1).expand(-1, -1, hid))
        # print(sampled_embeddings.shape)
        return sampled_embeddings

    def forward(self, input_ids, token_type_ids, attention_mask, neighbor_feature, center_lap_embedding = None, neighbor_lap_embedding = None):
        
        # compute lm embedding for center nodes
        lm = self.language_model.get_cls_embeddings(input_ids, token_type_ids, attention_mask)
        adaptor_embedding = self.adaptor(lm) 

        # map to gtf hidden space by a linear layer
        center_lin = self.embedding_h(adaptor_embedding)
        # center_pos_lin = self.embedding_lap_pos_enc(center_lap_embedding)
        center_emb = center_lin 

        neib_lin = self.embedding_h(neighbor_feature)
        # neib_pos_lin = self.embedding_lap_pos_enc(neighbor_lap_embedding)
        neib_emb = neib_lin

        # sampling
        neib_gtf = self.sample_neighbors(neib_lin, self.gtf_neib_num) # input for gtf, should include pe
        neib_loss = self.sample_neighbors(neib_lin, self.loss_neib_num) # for loss calculation, does not include pe

        seq = torch.cat((center_emb.unsqueeze(1), neib_gtf), dim = 1)
        
        for gtf in self.layers:
            h = gtf(seq)
        
        gtf_embedding = h[:, 0, :] # center emb
        gtf_neib_embedding = h[:, 1:, :] # neib emb

        # get graph-context mlm pred
        # if self.lm = 'masked':
        #     mlm_input = torch.Concatenate(gtf_embedding, token_emb)
        #     mlm_pred = self.mlm(mlm_input, corpus)
        #     return gtf_embedding, center_lin, neib_loss mlm_pred

        return gtf_embedding, center_lin, neib_loss 
    
    def mlm_loss(self, logits, labels):
        loss_fn = CrossEntropyLoss(ignore_index=-100)  # -100 indices are ignored in the loss computation
        active_loss = labels.view(-1) != -100  # Only compute loss for masked tokens
        reduced_logits = logits.view(-1, logits.size(-1))  # Flatten the logits
        reduced_labels = labels.view(-1)  # Flatten the labels

        # Calculate the loss only on the masked positions
        loss = loss_fn(reduced_logits[active_loss], reduced_labels[active_loss])
        
        return loss


    def generate_in_batch_negatives(self, input_tensor):

        b, _, hid = input_tensor.shape
        expanded_input = input_tensor.expand(b, b, hid)
        mask = ~torch.eye(b, dtype=torch.bool)
        mask = mask.unsqueeze(-1).expand(b, b, hid)
        negatives = expanded_input[mask].view(b, b-1, hid)
        
        return negatives # [b, b-1, hid]

    
    def contrastive_loss_node(self, gtf_embedding, neib_lin, tau=0.1):
        # gtf_embedding, center_lin, neib_lin = outputs
        gtf_embedding = gtf_embedding.unsqueeze(1) 
        
        # in-batch negatives
        neg = self.generate_in_batch_negatives(gtf_embedding) # take other center gtf emebddings as negative 

        # loss
        pos_sim = torch.sum(gtf_embedding * neib_lin, dim=2) / tau # similar to neighbor lm
        neg_sim = torch.matmul(gtf_embedding, neg.transpose(1, 2)) / tau   # dissimilar to other center gtf

        logits = torch.cat((pos_sim.unsqueeze(1), neg_sim), dim=2)
        probabilities = F.log_softmax(logits, dim=2)
        loss = -probabilities[:, 0].mean()  # Taking the mean of batch 

        return loss
    
    def contrastive_loss_modality(self, gtf_embedding, center_lin, tau = 0.1):
        # gtf_embedding, center_lin, neib_loss = outputs
        gtf_embedding = gtf_embedding.unsqueeze(1) # [b, 1, hid]
        center_lin = center_lin.unsqueeze(1)

        # in-batch negatives
        b_gtf = self.generate_in_batch_negatives(gtf_embedding) # [b, b-1, hid]
        b_lm = self.generate_in_batch_negatives(center_lin)

        # loss
        pos_sim = torch.sum(gtf_embedding * center_lin, dim=1) / tau # similar to lm embedding

        neg_sim_lm = torch.matmul(gtf_embedding, b_lm.transpose(1, 2)) / tau  # [b, 1, b-1]
        neg_sim_gtf = torch.matmul(center_lin, b_gtf.transpose(1, 2)) / tau  # [b, 1, b-1]
        neg_sim = torch.cat((neg_sim_lm, neg_sim_gtf), dim=2)  # [b, 1, 2*(b-1)]
        # neg_sim_lm = torch.matmul(gtf_embedding, neg.transpose(1, 2)) / tau   

        logits = torch.cat((pos_sim.unsqueeze(1), neg_sim), dim=2)
        probabilities = F.log_softmax(logits, dim=2)
        loss = -probabilities[:, 0].mean()  # Taking the mean of batch

        return loss


    def loss(self, outputs, tau=0.1, logits = None, labels = None):
        gtf_embedding, center_lin, neib_lin = outputs
        
        gtf_embedding = F.normalize(gtf_embedding)
        center_lin = F.normalize(center_lin)
        neib_lin = F.normalize(neib_lin)
        # output_normalized = (gtf_embedding, center_lin, neib_lin)

        loss_node = self.contrastive_loss_node(gtf_embedding, neib_lin, tau = 0.1)
        loss_modality = self.contrastive_loss_modality(gtf_embedding, center_lin, tau = 0.1)
        total_loss = loss_node + loss_modality

        # if self.lm = 'masked':
        #     mlm_loss = self.mlm_loss(logits, labels)
        #     total_loss = total_loss + mlm_loss
        
        return total_loss

