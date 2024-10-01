import os.path as osp

import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy
import ast
import random

import sys

class PredictShortestPathDataset(Dataset):
    def __init__(self, root, perfect_file="/raw/perfect.parquet", edge_index_file="/raw/edge_index.parquet", transform=None, target_transform=None):
        # Edge_index
        df = pd.read_parquet(root + edge_index_file, engine="auto")
        df = df.reset_index()
        for _, row in df.iterrows():
            self.edge_index = torch.from_numpy(numpy.array([row['Edge index'][0], row['Edge index'][1]]))
        print("Edge index has been generated")

        # X, Y, y_flag
        df = pd.read_parquet(root + perfect_file, engine="auto")
        df = df.reset_index()
        
        self.X = list()
        self.y = list()
        self.imperfect_y_flag = list()
        
        i = 1
        for _, row in df.iterrows():
            if( random.random()  < 0.5 ):
                continue

            elemX = torch.tensor(ast.literal_eval(row['X'].decode('utf-8')), dtype=torch.long)
            self.X.append(elemX)
            elemY = torch.tensor(ast.literal_eval(row['Y'].decode('utf-8'))[0], dtype=torch.long)
            self.y.append(elemY)
            elemFlag = torch.tensor(ast.literal_eval(row['Y'].decode('utf-8'))[1], dtype=torch.long)
            self.imperfect_y_flag.append(elemFlag)
            
            print(f"Sample #{i} has been generated")
            i+=1

        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return Data(x=self.X[idx], edge_index=self.edge_index, y=self.y[idx], imperfect_y_flag=self.imperfect_y_flag[idx], num_nodes=len(self.X[idx]))