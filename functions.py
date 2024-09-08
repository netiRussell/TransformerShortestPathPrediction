from torch.utils.data import BatchSampler, RandomSampler
from torch_geometric.loader import DataLoader
import torch
import os

def split_data(dataset, valid_ratio, total_samples, batch_size):
  train_size = int(total_samples * (1.0 - valid_ratio))
  validation_size = total_samples - train_size

  train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

  trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  validationLoader = DataLoader(validation_dataset, batch_size=100, shuffle=False)

  return trainLoader, validationLoader

def prepare_data(dataset, batch_size, valid_percantage):

  trainLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  sampler = RandomSampler(dataset, num_samples=int(len(dataset)*valid_percantage))
  validLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  return trainLoader, validLoader


def save_checkpoint(state, path='./savedGrads/checkpoint.pth.tar'):
    # Overwrite prev saving
    if os.path.isfile(path):
        os.remove(path)
    
    torch.save(state, path)


def is_correct(X, edge_list, path):
    source = (X == 5).nonzero(as_tuple=True)[0]
    destination = (X == 10).nonzero(as_tuple=True)[0].item()
    path = path.tolist()

    # Self-loop => no source case:
    if( len(source) == 0 ):
        if(path[-1] == destination and len(path) == 1):
            return 1
        else:
            return 0
    
    source = source.item()

    if( path[0] != source or path[-1] != destination ):
        return 0
    
    # Convert edge_list to a set for better perfomance
    edge_set = set(zip(edge_list[0].tolist(), edge_list[1].tolist()))
    
    # Check if each consecutive pair in the path exists in the edge set
    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) not in edge_set and (path[i + 1], path[i]) not in edge_set:
            return 0
    
    return 1