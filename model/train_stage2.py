import torch
from torch import nn
from adaglm import AdaGLM
from dl import NodeDataDataset, create_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm


def save_checkpoint(model, optimizer, epoch, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filename)
    print(f"Checkpoint saved: {filename}")

# Load the checkpoint
checkpoint_path = '/gpfsnyu/scratch/qz2086/AdaGLM/checkpoints/unmasked/checkpoint_10_cos_neighbor_noadaptor_90.pth'  # Update with the path to your last checkpoint

net_params = {
    'lm_dim': 768,  # Assuming the output dimension of the language model
    'hidden_dim': 768,
    'out_dim': 768,
    'n_classes': 3,
    'n_heads': 6,
    'in_feat_dropout': 0.1,
    'dropout': 0.1,
    'L': 12,  # Number of layers
    'readout': 'mean',
    'layer_norm': True,
    'batch_norm': True,
    'residual': True,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'lap_pos_enc': True,
    'pos_enc_dim': 768,  # Dimension of the Laplacian positional encoding
    'num_neighbors': 10,
    'gtf_neib_num': 7,
    'node_pos': 3,
    'lm_model': 'bert-base-uncased'
}

device = net_params['device']
model = AdaGLM(net_params).to(device)

data_dir = '/gpfsnyu/scratch/qz2086/AdaGLM/data'
domain = 'Economics'
tensor_file_path = f'{data_dir}/{domain}/tensor_data_10_neighbor_812.pt'
text_file_path = f'{data_dir}/{domain}/tokenized_text_abs_full.pt'

batch_size = 64
print('batch size:', batch_size)

dataloader = create_dataloader(tensor_file_path, text_file_path, batch_size, num_workers = 6, is_distributed = False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, min_lr = 1e-8, verbose=True)
scheduler = OneCycleLR(optimizer, max_lr=1e-5, steps_per_epoch=len(dataloader), epochs=900)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch

def train_model(model, data_loader, optimizer, scheduler, device, start_epoch, epochs):
    model.train()
    scaler = GradScaler()
    max_grad_norm = 1.0
    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            center_lap_encodings = batch['lap_embedding'].to(device)
            neighbor_lm_embeddings = batch['neighbor_lm_embedding'].to(device)
            neighbor_lap_encodings = batch['neighbor_lap_embedding'].to(device)

            one_hop_embedding = batch['one_hop_embedding'].to(device)
            one_hop_mask = batch['one_hop_mask'].to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids, token_type_ids, attention_mask, neighbor_lm_embeddings, center_lap_encodings, neighbor_lap_encodings, one_hop_embedding, one_hop_mask)
                nodeloss, modeloss, loss = model.loss(outputs, tau=0.1)
                
            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            print('loss:', loss, flush=True)
            print('node loss:', nodeloss, flush=True)

        avg_loss = total_loss / len(data_loader)
        current_lr = scheduler.get_last_lr()[0]  # Get the last learning rate from the scheduler
        print(f'Epoch {epoch + 1}/{start_epoch + epochs}, Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}')
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            filename = f"/gpfsnyu/scratch/qz2086/AdaGLM/checkpoints/unmasked/stage2_checkpoint_10_cos_neighbor_noadaptor_{epoch}.pth"
            print(filename)
            save_checkpoint(model, optimizer, epoch, filename)




# Continue training for an additional 100 epochs
train_model(model, dataloader, optimizer, scheduler, device, start_epoch=start_epoch, epochs=300)


