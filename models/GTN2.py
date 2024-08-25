# Transformer portion is based on: https://github.com/hkproj/pytorch-transformer/blob/main/model.py
# Improved and GCN added by Ruslan Abdulin

import torch
import torch.nn as nn

import math

# TODO: delete all (!) after done with development

# ! node_size = vocab_size
# InputEmbedder class takes in a sequenece and returns embedded values - vectors of d_model dimension
class InputEmbedder(nn.Module):
  def __init__( self, d_model: int, num_nodes: int ):
    super().__init__()
    self.d_model = d_model
    self.num_nodes = num_nodes
    self.embeddingLayer = nn.Embedding(num_nodes, d_model)
  
  def forward( self, input ):
    # input: (batch, max_path_len) --> (batch, max_path_len, d_model)
    return self.embeddingLayer(input) * math.sqrt(self.d_model)


# ! max_path_len = seq_len
# PoistionalEncoder generates vectors to be added to embedding vectors, so that each path has unique schema
class PoistionalEncoder(nn.Module):
  def __init__( self, d_model: int, max_path_len: int, dropout: float ):
    super().__init__()
    self.d_model = d_model
    self.max_path_len = max_path_len
    self.dropoutLayer = nn.Dropout(dropout)

    # Matrix of shape (max_path_len, d_model) to cover all embedding vectors
    pos_enc = torch.zeros(max_path_len, d_model, requires_grad=False)

    # 0 to max_path_len-1 Vector of shape (max_path_len, 1)
    position = torch.arange(0, max_path_len, dtype=torch.float).unsqueeze(1)

    # Division term from the paper
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    # Apply sin to even positions (every vector, every even value from embedding vector)
    pos_enc[:, 0::2] = torch.sin(position * div_term)

    # Apply cos to odd positions (every vector, every odd value from embedding vector)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    # Modify pos_enc shape to suit the mini-batch training pattern
    pos_enc = pos_enc.unsqueeze(0) # (1, max_path_len, d_model)

    # Cache the positional encodings, so that it is retrievable via model.state_dict()
    self.register_buffer('pos_enc', pos_enc)
  
  def forward( self, input ):
    input = input + self.pos_enc[:, :input.shape[1], :]
    return self.dropout(input)


# ! gamma = bias
# Normalizator lowers values in a sequence without affecting the elements' ratio towards each other
class Normalizator(nn.Module):
  def __init__( self, eps: float = 10**-6 ):
    super().__init__()
    # Epsilon is needed to avoid large numbers and division by 0
    self.eps = eps

    # Learnable params
    self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
    self.gamma = nn.Parameter(torch.zeros(1)) # Added
  
  def forward( self, input ):
    # input: (batch, seq_len, hidden_size)
    mean = input.mean(dim = -1, keepdim = True)
    std = input.std(dim = -1, keepdim = True) 

    # Formula from the paper
    return self.alpha * (input - mean) / (std + self.eps) + self.gamma


# FeedForwardBlock layer that makes use of two NNs (can be modified/reused for improvement)
class FeedForwardBlock(nn.Module):
  def __init__( self, d_model: int, d_ff: int, dropout: float ):
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_ff) # In the paper: W1, B1
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ff, d_model) # In the paper: W2, B2
  
  def forward( self, input):
    # input: (batch, max_path_len, d_model) --> (batch, max_path_len, d_ff) --> (batch, max_path_len, d_model)
    # Formula from the paper
    return self.linear2(self.dropout(torch.relu(self.linear1(input))))