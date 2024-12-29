# This file helps to eval the model online

import torch
import torch.nn as nn
from models.GTN2 import transformer_builder

from visualization import visualizeGraph, visualizeLoss
from data.dataset import PredictShortestPathDataset
from functions import get_data_subset, prepare_data, save_checkpoint, is_correct, generate_enc_mas

# -- Device --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Chosen device for training: {device}')


# -- Initial setup --
torch.set_printoptions(threshold=10000)

config = {
  "num_nodes": 2500
}

dataset = PredictShortestPathDataset(root="./data")
data = next(iter(get_data_subset(dataset, batch_size=1, n_samples=1)))[0]
edge_index_sample = data.edge_index.to(device)
print(f"Suggested path: {data.y}")
edge_set_sample = set(zip(edge_index_sample[0].tolist(), edge_index_sample[1].tolist()))

X = torch.tensor([[0]] * config['num_nodes']).to(device)


# -- Prompt mode --
mode = input("1 - Check if a path is connected\n2 - Find optimal path\nWhat would you like to do: ")


# -- Evaluation --
if mode == '1':
  print("\nCheck if a path is connected ----------------")
  path = input("Path with spaces between node ids: ")
  src = input("Source: ")
  dest = input("Destination: ")

  X[int(src)] = torch.tensor([5], dtype=torch.long)
  X[int(dest)] = torch.tensor([10], dtype=torch.long)

  path = [int(i) for i in path.split()]
  path = torch.tensor(path, dtype=torch.long)
  path.to(device)

  if is_correct(X, edge_set_sample, path) == 1:
    print("The path is Connected")
  else:
    print("The path is Disconnected")
  

elif mode == '2':
  # -- Model & Optimizer & Criterion --
  # checkpoint, model = transformer_builder( src_num_nodes=config['num_nodes'], tgt_num_nodes=config['num_nodes'], max_src_len=config['num_nodes']+1, max_tgt_len=config['num_nodes']+1, d_model=512, num_encoderBlocks=6, num_attnHeads=8, dropout=0.1, d_ff=2048, resume=True )
  # model.to(device)
  
  checkpoint, model = transformer_builder( max_src_len=config['num_nodes']+1, max_tgt_len=config['num_nodes']+1, d_model=774, num_encoderBlocks=7, num_attnHeads=9, dropout=0.1, d_ff=3072, resume=True, device=device )
  model.to(device)

  # -- Load model & optimizer --
  # if ( True ):
  #   checkpoint = torch.load('./savedGrads/checkpoint.pth.tar')
  #   model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  print("\nFind optimal path ------------------------")
  src = input("Source: ")
  dest = input("Destination: ")

  X[int(src)] = torch.tensor([5], dtype=torch.long).to(device)
  X[int(dest)] = torch.tensor([10], dtype=torch.long).to(device)

  encoder_mask = generate_enc_mas(num_nodes=config['num_nodes'], edge_set=edge_set_sample).to(device)

  prediction, _ = model( X, None, edge_index_sample, encoder_mask, config['num_nodes']+1)

  print(prediction)

else:
  print("Invalid input")


