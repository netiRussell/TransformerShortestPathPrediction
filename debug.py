import torch
import torch.nn as nn
from models.GTN2 import transformer_builder
from torch.optim.lr_scheduler import LambdaLR

from visualization import visualizeGraph, visualizeLoss
from data.dataset import PredictShortestPathDataset
from functions import get_data_subset, get_eval_subset, save_checkpoint, is_correct, generate_enc_mas, validate_curr_epoch



import sys # TODO: delete after done with development

# TODO: Compare results of batch_size = 20 and = 100. Try different hyperparameters !
# TODO: Try to retrain the model. Record outcome

# TODO: Post-training Q/A refining(find incorrect answers and use loss function with optimizer to nodge the weights)
# TODO: Try to eval the model. Record outcome

# TODO: MAYBE: Training dataset is too small and doesn’t cover all the optimal paths
# TODO: Try to run the model. Record outcome

# TODO: MAYBE: For the linear part after decoder before softmax, change dimensions of linear NN to 4.
# TODO: Try to run the model. Record outcome

# -- Device --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Chosen device for training: {device}')


# -- Config --
config = {
  "batch_size": 50,
  "num_epochs": 6,
  # TODO: start with bigger one and gradually go down to 10**-4 or some other small number
  "lr": 10**-4,
  "num_nodes": 100,
  "num_samples": 12500
}


# -- Dataset --
dataset = PredictShortestPathDataset(root="./data")


# -- Model & Optimizer & Criterion --
checkpoint, model = transformer_builder( max_src_len=config['num_nodes']+1, max_tgt_len=config['num_nodes']+1, d_model=512, num_encoderBlocks=6, num_attnHeads=8, dropout=0.1, d_ff=2048, resume=False, device=device )
model.to(device)

currTimeStep = 1
Twarmup = 10 # warmup will take place during the first 10 mini-batches
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']*currTimeStep/Twarmup, eps=1e-9)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)


# -- Training Loop --
model.train()
losses = list()

# Validation / Evaluation variables
validLoss = []

# Getting universal edge index
edge_index_sample = next(iter(get_data_subset(dataset, batch_size=1, n_samples=1)))[0].edge_index.to(device)
edge_set_sample = set(zip(edge_index_sample[0].tolist(), edge_index_sample[1].tolist()))

# Encoder mask
encoder_mask = generate_enc_mas(num_nodes=config['num_nodes'], edge_set=edge_set_sample).to(device)

for epoch in range(config['num_epochs']):
  # One epoch
  trainLoader = get_data_subset(dataset, batch_size=config['batch_size'], n_samples=config['num_samples'])

  for batch_index, batch in enumerate(trainLoader):
    # One batch
    optimizer.zero_grad()
    temp_losses = list()
    
    for i in range(config['batch_size']):
      # One sample
      
      # X
      encoder_input = batch[i].x.to(device)
      # Edge Index list
      adj_input = edge_index_sample
      
      # y, to be deleted after the model is trained
      decoder_input = batch[i].y.to(device)
      # y_flag (represents whether a sample is not an optimal path), to be deleted after the model is trained
      y_flag = batch[i].imperfect_y_flag.item()

    print(f"Epoch: {epoch+1}, Batch: {batch_index}")