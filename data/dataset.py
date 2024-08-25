import os.path as osp

import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import ast

class PredictShortestPathDataset(Dataset):
  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
      super().__init__(root, transform, pre_transform, pre_filter)

  @property
  def raw_file_names(self):
      return "perfect.csv"

  @property
  def processed_file_names(self):
    self.df = pd.read_csv(self.raw_paths[0])
    #return ['data_{}.pt'.format(i) for i in range(len(self.df)) ]
    return 'none.pt'

  def download(self):
      pass

  def process(self):
    # ID for corresponding dataset 
    idx = 0

    # Read the csv file
    self.df = pd.read_csv(self.raw_paths[0])
    self.df = self.df.reset_index()

    # For each row, create data and increment idx
    for index, row in self.df.iterrows():
        # Parameters for a dataset
        X = torch.tensor(ast.literal_eval(row['X']), dtype=torch.long)
        y = torch.tensor(ast.literal_eval(row['Y'])[0], dtype=torch.long).unsqueeze(1)
        imperfect_y_flag = torch.tensor(ast.literal_eval(row['Y'])[1], dtype=torch.long).unsqueeze(1)
        edge_index = torch.tensor(ast.literal_eval(row['Edge index']), dtype=torch.long)
        
        data = Data(x=X, edge_index=edge_index, y=y, imperfect_y_flag=imperfect_y_flag, num_nodes=len(X))

        torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
        idx += 1

  def len(self):
      return len(self.processed_file_names)

  def get(self, idx):
      data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
      return data
  
# dataset = set of generated data.
# When dataset[some index] is accesed, self.get function called
# This function retrieves corresponding file from the data folder
# dataset = PredictShortestPathDataset(root="./data")

# print(dataset[0].x)
# print(dataset[0].y)
# print(dataset[0].edge_index.t())