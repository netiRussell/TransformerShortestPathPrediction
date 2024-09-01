import torch
import torch.nn as nn
from models.GTN2 import transformer_builder

from visualization import visualizeGraph, visualizeLoss
from data.dataset import PredictShortestPathDataset
from functions import prepare_data, save_checkpoint

import sys # TODO: delete after done with development

# TODO: Training dataset is too small and doesn’t cover all the optimal paths
# TODO: For the linear part after decoder before softmax, change dimensions of linear NN to 4.

# TODO: Try to run the model. Record outcome

#TODO: Apply masks in encoder based on the adjacency matrix to avoid jumping from nodes that are not connected

# TODO: Try to run the model. Record outcome

# TODO: Compare results of batch_size = 20 and = 100. Record outcome

# -- Device --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Chosen device for training: {device}')


# -- Config --
config = {
  "batch_size": 50,
  "num_epochs": 6,
  # TODO: start with bigger one and gradually go down to 10**-4 or some other small number
  "lr": 10**-4,
  "num_nodes": 100
}


# -- Dataset --
dataset = PredictShortestPathDataset(root="./data")
total_samples = len(dataset)

trainLoader, validLoader = prepare_data( dataset=dataset, batch_size=config['batch_size'], valid_percantage=0.3)


# -- Visualize a single data sample --
visualizeGraph(dataset, num_nodes=100, run=False)


# -- Model & Optimizer & Criterion --

# max_src_len - max path length for source including EOS to start and end with
# max_tgt_len - max path length for tgt including EOS to start with
checkpoint, model = transformer_builder( src_num_nodes=config['num_nodes'], tgt_num_nodes=config['num_nodes'], max_src_len=config['num_nodes']+1, max_tgt_len=config['num_nodes']+1, d_model=512, num_encoderBlocks=6, num_attnHeads=8, dropout=0.1, d_ff=2048, resume=False )
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

# TODO: make sure loss doesn't count EOS by utilizing ignore_index param
criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)


# -- Training Loop --
model.train()

# mask relations between nodes that are not neighbors
encoder_mask = None # (Batch, 1, 1, Seq_Len)

# mask tokens
decoder_mask = None # (Batch, 1, Seq_Len, Seq_Len)
#decoder_mask = torch.triu(torch.ones(1, config['num_nodes']+1, config['num_nodes']+1), diagonal=1).type(torch.int) == 0 #(1, Seq_Len, Seq_Len)

for epoch in range(config['num_epochs']):
  for batch_index, batch in enumerate(trainLoader):
    optimizer.zero_grad()
    
    for i in range(config['batch_size']):
      # X
      encoder_input = batch[i].x.to(device) # (Batch, Seq_Length)
      # Edge Index list
      adj_input = batch[i].edge_index.to(device)
      
      # TODO: undesrtand what is decoder_input in the source code
      # y, to be deleted after the model is trained
      decoder_input = batch[i].y.to(device) # (Batch, Seq_Length)

      # TODO: create forward method for the model
      encoder_output = model.encode( encoder_input, adj_input, src_mask=encoder_mask )
      decoder_output = model.decode( encoder_output, encoder_mask, decoder_input, decoder_mask )
      proj_output = model.project( decoder_output )
      
      # Loss, to be deleted after the model is trained
      loss = criterion(proj_output, decoder_input)
      loss.backward()

      
      break
    
    optimizer.step()

    break
  break


# TODO: implement evaluation
