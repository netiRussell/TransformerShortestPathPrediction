import os.path as osp

import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy
import ast
import h5py

import sys

class PredictShortestPathDataset(Dataset):
  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
      super().__init__(root, transform, pre_transform, pre_filter)

  @property
  def raw_file_names(self):
      return ["edge_index.parquet", "perfect.parquet"]

  @property
  def processed_file_names(self):
    #return 'none.pt'

    self.df = pd.read_parquet(self.raw_paths[1])
    return ['data_{}.pt'.format(i) for i in range(len(self.df)) ]
    

  def download(self):
      pass

  def process(self):
    # ID for corresponding dataset 
    idx = 0

    # Read and retrieve edge_index
    self.df = pd.read_parquet(self.raw_paths[0], engine="auto")
    self.df = self.df.reset_index()

    for _, row in self.df.iterrows():
        edge_index = torch.from_numpy(numpy.array([row['Edge index'][0], row['Edge index'][1]]))

    # Read the parquet file
    self.df = pd.read_parquet(self.raw_paths[1], engine="auto")
    self.df = self.df.reset_index()

    # For each row, create data and increment idx
    for _, row in self.df.iterrows():
        # Parameters for a dataset
        X = torch.tensor(ast.literal_eval(row['X'].decode('utf-8')), dtype=torch.long)
        y = torch.tensor(ast.literal_eval(row['Y'].decode('utf-8'))[0], dtype=torch.long)
        imperfect_y_flag = torch.tensor(ast.literal_eval(row['Y'].decode('utf-8'))[1], dtype=torch.long).unsqueeze(1)
        
        data = Data(x=X, edge_index=edge_index, y=y, imperfect_y_flag=imperfect_y_flag, num_nodes=len(X))

        torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))

        print(f'data_{idx}.pt is generated')
        idx += 1

  def len(self):
      return len(self.processed_file_names)

  def get(self, idx):
      data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
      return data
  
# TODO:
# - Take edge_index out and save it once
# - Try to use HDF5 file format instead of pt (you can save and load everythin on your own. Just return Data in get method)
# - Consider saving each a batch in each file instead of a single sample.