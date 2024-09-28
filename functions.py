from torch.utils.data import BatchSampler, RandomSampler
from torch_geometric.loader import DataLoader
import torch
import os

def prepare_data(dataset, batch_size, n_epochs):

  trainLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  sampler = RandomSampler(dataset, num_samples=len(dataset))
  valid_batch_size = int(len(dataset)/n_epochs)

  validLoader = DataLoader(dataset, batch_size=valid_batch_size, shuffle=False, sampler=sampler)

  print(f'Lenght of dataset: {len(dataset)}, batch_size for validation and eval: {valid_batch_size}')

  return trainLoader, validLoader

def get_data_subset(dataset, batch_size, n_samples):
    
    sampler = RandomSampler(dataset, num_samples=n_samples)
    trainLoader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return trainLoader

def get_eval_subset(dataset, valid_percantage):
  sampler = RandomSampler(dataset, num_samples=int(len(dataset)*valid_percantage))
  validLoader = DataLoader(dataset, batch_size=100, shuffle=False, sampler=sampler)

  return validLoader


def save_checkpoint(state, path='./savedGrads/checkpoint.pth.tar'):
    # Overwrite prev saving
    if os.path.isfile(path):
        os.remove(path)
    
    torch.save(state, path)


def is_correct(X, edge_set, path):
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
    
    # Check if each consecutive pair in the path exists in the edge set
    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) not in edge_set and (path[i + 1], path[i]) not in edge_set:
            return 0
    
    return 1


def generate_enc_mas(num_nodes, edge_set):
    tmp_mask = torch.zeros((num_nodes+1, num_nodes+1))

    # Check if each consecutive pair in the path exists in the edge set
    for consideredNode in range(tmp_mask.shape[0]):
        # Self loop case
        tmp_mask[consideredNode][consideredNode] = 1
        
        # All other connections
        for otherNode in range(tmp_mask.shape[0]):
            if (consideredNode, otherNode) in edge_set:
                tmp_mask[consideredNode][otherNode] = 1
    
    # EOS case
    initial_value_cm = list(1 for _ in range(num_nodes))
    initial_value_cm.append(0)
    tmp_mask[-1] = torch.tensor(initial_value_cm)

    return tmp_mask


# TODO: create a function that outputs current validaiton loss while using different untouched batch of data
def validate_curr_epoch( validIter, edge_index_sample, edge_set_sample, model, encoder_mask, config, device ):
    model.eval()

    try:
        # Get the next validation / evaluation batch
        batch = next(validIter)

        complete_success_rate = []

        for i in range(len(batch)):
            # Imperfect sample; to be disregarded
            # y_flag = batch[i].imperfect_y_flag.item()
            # if( y_flag == 1 ):
            #   continue

            # X
            encoder_input = batch[i].x.to(device)
            # Edge Index list
            adj_input = edge_index_sample
            # y
            label = batch[i].y.to(device)

            # Generate prediction (we're interested in steps, not the probs)
            prediction, _ = model( encoder_input, None, adj_input, encoder_mask, config['num_nodes']+1)

            # Check if the length of the output is correct
            if(len(label) != len(prediction)):
                complete_success_rate.append(0)
                continue
            
            # Check if all the nodes are correct and src and dest are correct
            complete_success_rate.append( is_correct(encoder_input, edge_set_sample, prediction) )
        
        loss = (sum(complete_success_rate) / len(complete_success_rate))*100
        print(f"Current validation percantage: {loss}%")

        model.train()
        return loss
    
    except StopIteration:
        # Avoid raising StopIteration
        model.train()
        print("No more untouched data for valid/eval left")
