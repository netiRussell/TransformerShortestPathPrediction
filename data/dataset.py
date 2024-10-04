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
    # return 'none.pt'

    self.df = pd.read_parquet(self.raw_paths[1])
    return ['data_{}.h5'.format(i) for i in range(len(self.df)//(2*100)) ]
    

  def download(self):
    pass

  def process(self):
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
    idx = 0 # ID for corresponding sample
    counter = 0
    num_samples_per_file = 100
    h5f = None
    dt = h5py.vlen_dtype(numpy.dtype('long'))

    for _, row in self.df.iterrows():

        # Every even sample is disregarded to minimize the total # of samples
        if counter % 2 != 0:
           counter += 1
           continue

        if(idx % num_samples_per_file == 0):
            # Create a file and initialize datasets # TODO: init or declare?
            num_nodes = len(numpy.array(ast.literal_eval(row['X'].decode('utf-8'))))
            h5_file_path = osp.join(self.processed_dir, f'data_{idx // num_samples_per_file}.h5')
            h5f = h5py.File(h5_file_path, 'w')

            x_set = h5f.create_dataset('x', shape=(num_samples_per_file, num_nodes, ), dtype=dt)
            y_set = h5f.create_dataset('y', shape=(num_samples_per_file,), dtype=dt)
            flag_set = h5f.create_dataset('imperfect_y_flag', shape=(num_samples_per_file,), dtype=dt)
            h5f.attrs['num_nodes'] = num_nodes

        #Append samples
        row_num = idx % num_samples_per_file
        x_set[row_num, ...] = torch.tensor(ast.literal_eval(row['X'].decode('utf-8'))).numpy()
        y_set[row_num, ...] = torch.tensor(ast.literal_eval(row['Y'].decode('utf-8'))[0]).numpy()
        flag_set[row_num, ...] = torch.tensor([ast.literal_eval(row['Y'].decode('utf-8'))[1]]).numpy()

        idx += 1
        counter += 1
        print(f'data_{idx}.pt generated in {idx // num_samples_per_file}')

  def len(self):
        return len(self.processed_file_names) // 2 # since every 2nd sample is disregarded

  def get(self, idx):
        num_samples_per_file = 100

        h5_file_path = osp.join(self.processed_dir, 'edge_index.h5')
        with h5py.File(h5_file_path, 'r') as h5f:
            edge_index = torch.tensor(h5f['edge_index'][:])

        h5_file_path = osp.join(self.processed_dir, f'data_{idx // num_samples_per_file}.h5')
        h5f = h5py.File(h5_file_path, 'r')

        x_numeric = numpy.stack(h5f['x'][idx % num_samples_per_file]).astype(numpy.int64)
        X = torch.tensor(x_numeric)
        
        y = torch.tensor(h5f['y'][idx % num_samples_per_file])
        imperfect_y_flag = torch.tensor(h5f['imperfect_y_flag'][idx % num_samples_per_file])
        num_nodes = h5f.attrs['num_nodes']

        data = Data(x=X, edge_index=edge_index, y=y, imperfect_y_flag=imperfect_y_flag, num_nodes=num_nodes)

        return data