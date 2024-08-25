import torch
import torch.nn as nn

# node_size = vocab_size
class Embeddings(nn.Module):
  def __init__(self, d_model: int, num_nodes: int):
    super().__init__()
    self.d_model = d_model
    self.num_nodes = num_nodes
