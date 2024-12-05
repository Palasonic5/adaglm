import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptor import AdaptorLayer
from gt_layer_self_attention import GraphTransformerLayer
from transformers import BertModel, BertTokenizer
from language_model import UnmaskedBERT, MaskedBERT
from lm_decoder import DecoderModel


class AdaGLM(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        lm_dim = net_params['lm_dim'] # node_dim (feat is an integer), original name  'in_dim'
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        num_neighbors = net_params['num_neighbors']
        pos_enc_dim = net_params['pos_enc_dim']

        self.lm = net_params['lm']
        # self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.num_neighbors = net_params['num_neighbors']
        self.gtf_neib_num = net_params['gtf_neib_num']

        self.device = net_params['device']

        self.loss_neib_num = net_params['node_pos']
        self.mlm_prob = net_params['mlm_prob']
        self.mode = net_params['mode']
        self.state = net_params['state']

        # self.mod_pos_num = net_params['mod_pos']
        # self.mod_neg_num = net_params['mod_neg']

        assert self.lm in ['unmasked', 'masked']

        if self.lm == 'unmasked':
            self.language_model = UnmaskedBERT(model_name = 'bert-base-uncased', state = self.state)
        elif self.lm == 'masked':
            self.language_model = MaskedBERT(model_name = 'bert-base-uncased', num_layers = 12, mlm_prob = self.mlm_prob, mode = self.mode)
        
        # layers
        self.adaptor = AdaptorLayer(self.state)
        
        # self.embedding_h = nn.Embedding(lm_dim, hidden_dim) # node feat is an integer
        self.embedding_h = nn.Linear(lm_dim, lm_dim)

        
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, lm_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList()
        self.layers.append(GraphTransformerLayer(in_dim=lm_dim, 
                                                out_dim=hidden_dim, 
                                                num_heads=num_heads,
                                                num_neighbor=self.gtf_neib_num, 
                                                dropout=0.1,
                                                layer_norm=True,
                                                batch_norm=False,
                                                residual=False,
                                                use_bias=True))
        self.layers.extend([GraphTransformerLayer(in_dim=hidden_dim, 
                                                out_dim=hidden_dim, 
                                                num_heads=num_heads,
                                                num_neighbor=self.gtf_neib_num, 
                                                dropout=0.1,
                                                layer_norm=True,
                                                batch_norm=False,
                                                residual=True,
                                                use_bias=True) for _ in range(n_layers-2)])
        self.layers.append(GraphTransformerLayer(in_dim=hidden_dim, 
                                                out_dim=out_dim, 
                                                num_heads=num_heads,
                                                num_neighbor=self.gtf_neib_num, 
                                                dropout=0.1,
                                                layer_norm=True,
                                                batch_norm=False,
                                                residual=False,
                                                use_bias=True))

        print('number of layers in modulelist:', len(self.layers))

        self.mode_loss_weight = net_params['mode_loss']
        self.tau = net_params['tau']

        self.use_adaptor = net_params['if_adaptor']
        self.use_gtf = net_params['if_gtf']


    def sample_neighbors(self, sequence_tensor, num_samples):
        b, seq, hid = sequence_tensor.shape
        # num_samples = self.gtf_neib_num

        if seq < num_samples:
            sampled_embeddings = sequence_tensor
            # raise ValueError(f"Not enough embeddings to sample from. Required: {num_samples}, Available: {seq}")
        
        # Create a tensor of indices for uniform sampling without replacement
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        indices = torch.multinomial(torch.ones((b, seq)), num_samples, replacement=False).to(device)

        # Gather the sampled embeddings using the generated indices
        sampled_embeddings = torch.gather(sequence_tensor, 1, indices.unsqueeze(-1).expand(-1, -1, hid))
        # print(sampled_embeddings.shape)
        return sampled_embeddings

    def forward(self, input_ids, token_type_ids, attention_mask, neighbor_feature, center_lap_embedding, neighbor_lap_embedding, one_hop_embedding, one_hop_mask):
        
        # compute lm embedding for center nodes
        if self.lm == 'masked':
            lm, last_layer_embeddings, labels = self.language_model(input_ids, token_type_ids, attention_mask)

        elif self.lm =='unmasked':
            lm = self.language_model.get_cls_embeddings(input_ids, token_type_ids, attention_mask)
            lm_loss = 0
        # lm = self.language_model.get_cls_embeddings(input_ids, token_type_ids, attention_mask)
        # lm = lm[:, -1, :]
        # lm = lm.squeeze(1)
        if self.use_adaptor:
            adaptor_embedding = self.adaptor(lm) 
        
        # map to gtf hidden space by a linear layer
        # center_lin = self.embedding_h(adaptor_embedding)
            center_lin = adaptor_embedding
        else:
            center_lin = lm[:, -1, :]
            center_lin = center_lin.squeeze(1)
        # print('center lin shape:', center_lin.shape, flush = True)

        center_pos_lin = self.embedding_lap_pos_enc(center_lap_embedding)
        # print('center position linear shape', center_pos_lin.shape, flush = True)
        center_emb = center_lin + center_pos_lin

        neib_lin = self.embedding_h(neighbor_feature)
        neib_pos_lin = self.embedding_lap_pos_enc(neighbor_lap_embedding)
        # print('neighbor position linear shape', neib_pos_lin.shape)
        neib_emb = neib_lin + neib_pos_lin

        one_hop_embedding = self.embedding_h(one_hop_embedding) # only reconstruct 1-hop neighbor in loss
        one_hop_embedding = one_hop_embedding * one_hop_mask.unsqueeze(-1)

        # sampling
        neib_gtf = self.sample_neighbors(neib_emb, self.gtf_neib_num) # input for gtf, should include pe
        neib_loss = self.sample_neighbors(one_hop_embedding, self.loss_neib_num) # for loss calculation, does not include pe

        seq = torch.cat((center_emb.unsqueeze(1), neib_gtf), dim = 1)
        if self.use_gtf:
        # gtf_embedding = center_emb
            for gtf in self.layers:
                seq = gtf(seq)
        
            gtf_embedding = seq[:, 0, :] # center emb
            gtf_neib_embedding = seq[:, 1:, :] # neib emb
        else:
            gtf_embedding = center_lin

        if self.lm == 'masked':
            modified_last_layer_embedding = last_layer_embeddings.clone()
            modified_last_layer_embedding[:, 0, :] = gtf_embedding

            lm_loss = self.language_model.mlm_loss(modified_last_layer_embedding, labels)
        elif self.lm == 'unmasked':
            lm_loss = 0
        # get graph-context mlm pred
        # if self.lm = 'masked':
        #     mlm_input = torch.Concatenate(gtf_embedding, token_emb)
        #     mlm_pred = self.mlm(mlm_input, corpus)
        #     return gtf_embedding, center_lin, neib_loss mlm_pred

        return gtf_embedding, center_lin, neib_loss, lm_loss
    



    def generate_in_batch_negatives(self, input_tensor):
        b, _, hid = input_tensor.shape
        expanded_input = input_tensor.expand(b, b, hid)
        mask = ~torch.eye(b, dtype=torch.bool)
        mask = mask.unsqueeze(-1).expand(b, b, hid)
        negatives = expanded_input[mask].view(b, b-1, hid)
        
        return negatives # [b, b-1, hid]

    def contrastive_loss_node_new(self, gtf_embedding, neib_lin, tau=0.5):
        gtf_embedding = gtf_embedding.unsqueeze(1) 
        
        # in-batch negatives
        neg = self.generate_in_batch_negatives(gtf_embedding) # take other center gtf emebddings as negative 
        pos_sim = F.cosine_similarity(gtf_embedding, neib_lin, dim=2)
        pos_sim = torch.exp(pos_sim) / tau

        neg_sim = F.cosine_similarity(gtf_embedding, neg, dim=2)
        neg_sim = torch.exp(neg_sim) / tau

        pos_sum = pos_sim.sum(dim=1)
        neg_sum = neg_sim.sum(dim=1)

        loss = -torch.log(pos_sum / (neg_sum + 1e-8)).mean()

        return loss

    
    def contrastive_loss_node(self, gtf_embedding, neib_lin, tau=0.5):
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

    def contrastive_loss_modality_new(self, gtf_embedding, center_lin, tau = 0.5):
        gtf_embedding = gtf_embedding.unsqueeze(1) # [b, 1, hid]
        center_lin = center_lin.unsqueeze(1)

        # in-batch negatives
        b_gtf = self.generate_in_batch_negatives(gtf_embedding) # [b, b-1, hid]
        b_lm = self.generate_in_batch_negatives(center_lin)

        pos_sim = F.cosine_similarity(gtf_embedding, center_lin, dim=2)
        pos_sim = torch.exp(pos_sim) / tau

        neg_sim = F.cosine_similarity(gtf_embedding, b_lm, dim=2)
        neg_sim = torch.exp(neg_sim) / tau

        pos_sum = pos_sim.sum(dim=1)
        neg_sum = neg_sim.sum(dim=1)

        # Compute the InfoNCE loss
        loss = -torch.log(pos_sum / (neg_sum + 1e-8)).mean()

        return loss



    
    def contrastive_loss_modality(self, gtf_embedding, center_lin, tau = 0.5):
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
        # neg_sim = torch.cat((neg_sim_lm, neg_sim_gtf), dim=2)  # [b, 1, 2*(b-1)]
        # neg_sim_lm = torch.matmul(gtf_embedding, neg.transpose(1, 2)) / tau   

        logits = torch.cat((pos_sim.unsqueeze(1), neg_sim_lm), dim=2)
        probabilities = F.log_softmax(logits, dim=2)
        loss = -probabilities[:, 0].mean()  # Taking the mean of batch

        return loss


    def loss(self, outputs, tau=0.5, logits = None, labels = None):
        gtf_embedding, center_lin, neib_lin, lm_loss = outputs

        loss_node = self.contrastive_loss_node_new(gtf_embedding, neib_lin, tau = self.tau)
        loss_modality = self.contrastive_loss_modality_new(gtf_embedding, center_lin, tau = self.tau)
        # loss_modality = 0
        
        loss_node_weight = 1 - self.mode_loss_weight
        total_loss = loss_node_weight * loss_node + self.mode_loss_weight * loss_modality + lm_loss

        # if self.lm = 'masked':
        #     mlm_loss = self.mlm_loss(logits, labels)
        #     total_loss = total_loss + mlm_loss
        
        return lm_loss, loss_node, loss_modality, total_loss


    


    # def mlm_loss(self, logits, labels):
    #     loss_fn = CrossEntropyLoss(ignore_index=-100)  # -100 indices are ignored in the loss computation
    #     active_loss = labels.view(-1) != -100  # Only compute loss for masked tokens
    #     reduced_logits = logits.view(-1, logits.size(-1))  # Flatten the logits
    #     reduced_labels = labels.view(-1)  # Flatten the labels

    #     # Calculate the loss only on the masked positions
    #     loss = loss_fn(reduced_logits[active_loss], reduced_labels[active_loss])
        
    #     return loss