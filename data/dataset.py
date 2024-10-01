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
      self.edge_index = None

  @property
  def raw_file_names(self):
      return ["edge_index.parquet", "perfect.parquet"]

  @property
  def processed_file_names(self):
    #return 'none.pt'

    self.df = pd.read_parquet(self.raw_paths[1])
    return ['data_{}.h5'.format(i) for i in range(len(self.df)) ]
    

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
    
    # Save edge_index
    h5_file_path = osp.join(self.processed_dir, 'edge_index.h5')
    with h5py.File(h5_file_path, 'w') as h5f:
        h5f.create_dataset('edge_index', data=edge_index.numpy())

    # Read the parquet file
    self.df = pd.read_parquet(self.raw_paths[1], engine="auto")
    self.df = self.df.reset_index()

    
    # For each row, create data and increment idx
    for _, row in self.df.iterrows():
        # Parameters for a dataset
        X = torch.tensor(ast.literal_eval(row['X'].decode('utf-8')), dtype=torch.long)
        y = torch.tensor(ast.literal_eval(row['Y'].decode('utf-8'))[0], dtype=torch.long)
        imperfect_y_flag = torch.tensor(ast.literal_eval(row['Y'].decode('utf-8'))[1], dtype=torch.long).unsqueeze(1)

        h5_file_path = osp.join(self.processed_dir, f'data_{idx}.h5')
        with h5py.File(h5_file_path, 'w') as h5f:
            # Save each dataset sample into the HDF5 file
            group = h5f.create_group(f'data_{idx}')
            group.create_dataset('x', data=X.numpy())
            group.create_dataset('y', data=y.numpy())
            group.create_dataset('imperfect_y_flag', data=imperfect_y_flag.numpy())
            group.attrs['num_nodes'] = len(X)

            print(f'data_{idx}.pt is generated')
            idx += 1

  def len(self):
      return len(self.processed_file_names)

  def get(self, idx):
      h5_file_path = osp.join(self.processed_dir, 'edge_index.h5')
      with h5py.File(h5_file_path, 'r') as h5f:
         edge_index = torch.tensor(h5f['edge_index'][:])

      h5_file_path = osp.join(self.processed_dir, f'data_{idx}.h5')
      with h5py.File(h5_file_path, 'r') as h5f:
        group = h5f[f'data_{idx}']
        X = torch.tensor(group['x'][:])
        y = torch.tensor(group['y'][:])
        imperfect_y_flag = torch.tensor(group['imperfect_y_flag'][:])
        num_nodes = group.attrs['num_nodes']

      data = Data(x=X, edge_index=edge_index, y=y, imperfect_y_flag=imperfect_y_flag, num_nodes=num_nodes)
      return data
  
# TODO:
# - Take edge_index out and save it once
# - Try to use HDF5 file format instead of pt (you can save and load everythin on your own. Just return Data in get method)

# - Consider saving each a batch in each file instead of a single sample.