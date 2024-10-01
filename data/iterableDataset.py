import os.path as osp

import torch
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pandas as pd
import numpy
import ast
import random

import sys

class PredictShortestPathDataset(IterableDataset):
    def __init__(self, root, randomness_rate, perfect_file="/raw/perfect.parquet", edge_index_file="/raw/edge_index.parquet", transform=None, target_transform=None):
        self.root = root
        self.perfect_file = perfect_file
        self.randomness_rate = randomness_rate

        # Edge_index
        df = pd.read_parquet(self.root + edge_index_file, engine="auto")
        df = df.reset_index()
        for _, row in df.iterrows():
            self.edge_index = torch.from_numpy(numpy.array([row['Edge index'][0], row['Edge index'][1]]))
        print("Edge index has been generated")
    
    def generate(self):

        # X, Y, y_flag
        df = pd.read_parquet(self.root + self.perfect_file, engine="auto")
        df = df.reset_index()
            
        for _, row in df.iterrows():
            if( random.random()  < self.randomness_rate ):
                continue

            elemX = torch.tensor(ast.literal_eval(row['X'].decode('utf-8')), dtype=torch.long)
            elemY = torch.tensor(ast.literal_eval(row['Y'].decode('utf-8'))[0], dtype=torch.long)
            elemFlag = torch.tensor(ast.literal_eval(row['Y'].decode('utf-8'))[1], dtype=torch.long)

            yield Data(x=elemX, edge_index=self.edge_index, y=elemY, imperfect_y_flag=elemFlag, num_nodes=len(elemX))

    def __iter__(self):
        return iter(self.generate())
    
    def __getitem__(self, idx):
        return Data(x=self.X[idx], edge_index=self.edge_index, y=self.y[idx], imperfect_y_flag=self.imperfect_y_flag[idx], num_nodes=len(self.X[idx]))