import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.GTN2 import transformer_builder

from visualization import visualizeGraph, visualizeLoss
from data.dataset import PredictShortestPathDataset
from functions import prepare_data, save_checkpoint

# TODO: Training dataset is too small and doesn’t cover all the optimal paths
# TODO: For the linear part after decoder before softmax, change dimensions of linear NN to 4.

# TODO: Try to run the model. Record outcome

#TODO: Apply masks in encoder based on the adjacency matrix to avoid jumping from nodes that are not connected

# TODO: Try to run the model. Record outcome

# TODO: Compare results of num_epochs = 20 and = 100. Record outcome

# -- Device --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Chosen device for training: {device}')


# -- Config --
config = {
  "batch_size": 6,
  "num_epochs": 50,
  # TODO: start with bigger one and gradually go down to 10**-4 or some other small number
  "lr": 10**-4,
  "num_nodes": 100
}


# -- Dataset --
dataset = PredictShortestPathDataset(root="./data")
total_samples = len(dataset)

trainLoader, validLoader = prepare_data( dataset=dataset, batch_size=config['batch_size'], valid_percantage=0.3)

summaryWriter = SummaryWriter('summary')


# -- Visualize a single data sample --
visualizeGraph(dataset, num_nodes=100, run=False)


# -- Model & Optimizer & Criterion --

# max_src_len - max path length for source including EOS to start and end with
# max_tgt_len - max path length for tgt including EOS to start with
model = transformer_builder( src_num_nodes=config['num_nodes'], tgt_num_nodes=config['num_nodes'], max_src_len=config['num_nodes']+2, max_tgt_len=config['num_nodes']+1, d_model=512, num_encoderBlocks=6, num_attnHeads=8, dropout=0.1, d_ff=2048, resume=False ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

# TODO: make sure loss doesn't count EOS by utilizing ignore_index param
criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)





