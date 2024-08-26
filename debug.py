from models.GTN2 import GCNs
from functions import prepare_data
from data.dataset import PredictShortestPathDataset
import torch

dataset = PredictShortestPathDataset(root="./data")
total_samples = len(dataset)

trainLoader, validLoader = prepare_data( dataset=dataset, batch_size=50, valid_percantage=0.3)

gcn = GCNs(0.2)
gcn.train()

for batch in trainLoader:
  for i in range(3):
    x = batch[i].x.permute(1,0)
    y = torch.cat(( batch[i].y.permute(1,0), torch.tensor([[len(batch[i].x)]]) ), 1)

    print(x)

    print(gcn(x, batch[i].edge_index))

    print("\n\n")
  break