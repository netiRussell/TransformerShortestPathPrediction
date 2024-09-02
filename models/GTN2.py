# Transformer portion is based on: https://github.com/hkproj/pytorch-transformer/blob/main/model.py
# Improved and GCN added by Ruslan Abdulin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv

import math

import sys # TODO: delete after done with development

# TODO: delete all (!) after done with development
# Must be in eval mode to work properly after training.

# Custom Sigmoid for final GCN output to fit transformer
class CustomSigmoid(nn.Module):
  def __init__(self):
    super(CustomSigmoid, self).__init__()

  def forward(self, x, beta=1):
    return len(x)*torch.sigmoid(x)


# -- GCN layer --
class GCNs(nn.Module):
  def __init__( self, dropout: int ):
    super().__init__()
    # num GCN layers = sqrt(n_nodes) - 1
    self.gcn1 = GCN2Conv(1, 0.3)
    self.gcn2 = GCN2Conv(1, 0.3)
    self.gcn3 = GCN2Conv(1, 0.3)
    self.gcn4 = GCN2Conv(1, 0.3)
    self.gcn5 = GCN2Conv(1, 0.3)
    self.gcn6 = GCN2Conv(1, 0.3)
    self.gcn7 = GCN2Conv(1, 0.3)
    self.gcn8 = GCN2Conv(1, 0.3)
    self.gcn9 = GCN2Conv(1, 0.3)

    self.dropout = nn.Dropout(dropout)
    self.sigmoidNumNod = CustomSigmoid()
  
  def forward( self, input, adj ):
    x_0 = input.float()

    out = self.gcn1(x_0, x_0, adj)
    out = self.dropout(out)

    out = F.leaky_relu(self.gcn2(out, x_0, adj))
    out = self.dropout(out)

    out = F.relu(self.gcn3(out, x_0, adj))
    out = self.dropout(out)

    out = F.relu(self.gcn4(out, x_0, adj))
    out = self.dropout(out)

    out = torch.sigmoid((self.gcn5(out, x_0, adj)))
    out = self.dropout(out)

    out = F.relu(self.gcn6(out, x_0, adj))
    out = self.dropout(out)

    out = F.relu(self.gcn7(out, x_0, adj))
    out = self.dropout(out)

    out = F.relu(self.gcn8(out, x_0, adj))
    out = self.dropout(out)

    out = self.sigmoidNumNod(self.gcn9(out, x_0, adj))
    out = out.squeeze(1).long()

    return out

# ! node_size = vocab_size
# InputEmbedder class takes in a sequenece and returns embedded values - vectors of d_model dimension
class InputEmbedder(nn.Module):
  def __init__( self, d_model: int, num_nodes: int ):
    super().__init__()
    self.d_model = d_model
    self.num_nodes = num_nodes+1 # + 1 for EOS
    self.embeddingLayer = nn.Embedding(self.num_nodes, d_model)
  
  def forward( self, input ):
    # input: (max_path_len) --> (max_path_len, d_model)
    return self.embeddingLayer(input) * math.sqrt(self.d_model)


# ! max_path_len = seq_len
# PoistionalEncoder generates vectors to be added to embedding vectors, so that each path has unique schema
class PoistionalEncoder(nn.Module):
  def __init__( self, d_model: int, max_path_len: int, dropout: float ):
    super().__init__()
    self.d_model = d_model
    self.max_path_len = max_path_len
    self.dropout = nn.Dropout(dropout)

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

    # Cache the positional encodings, so that it is retrievable via model.state_dict()
    self.register_buffer('pos_enc', pos_enc)
  
  def forward( self, input ):
    input = input + self.pos_enc[ :input.shape[0], :]
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


# ! num_heads = h
# MultiHeadAttentionBlock calculates relations between elements in a sequence
# TODO: consider step softmax(...), in ... we calculate matrix that allows us to omit(mask) relations between some elements. Check if it is possible to write matrix that utilizes this omit-feature based on the given adjacency matrix. If it is possible, then implement one. This way, I would have a matrix that prevents jumping from node A to node B that is not directly connected to node A.
  # https://www.youtube.com/watch?v=ISNdQcPhsts @ minute 29
class MultiHeadAttentionBlock(nn.Module):
  def __init__( self, d_model: int, num_heads: int, dropout: float ):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads

    # Floor division to get d_k = number of embedding values per head
    assert d_model % num_heads == 0, "d_model is not divisible by num_heads"
    self.d_k = d_model // num_heads

    # Weights for finding q, k, v. 
    # To find q/k/v with weights = pos_enc(embedding(input))*self.w_q/self.w_k/self.w_v
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)

    # Weights applied after concatination of all heads
    self.w_o = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)
  
  @staticmethod # no need for an instance, can be called directly from MultiHeadAttentionBlock
  def calcAttention( query, key, value, mask, dropout: nn.Dropout ):
    d_k = query.shape[-1]

    # Formula from the paper
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask to omit relations between certain nodes
    if mask is not None:
      # All corresponding values of mask that are equal 0 will be replaced by -1e9 in attention_scores
      attention_scores.masked_fill_(mask == 0, -1e9)
    
    # Softmax
    attention_scores = attention_scores.softmax(dim=-1)

    # Dropout
    if dropout is not None:
      attention_scores = dropout(attention_scores)
    
    # Formula continues
    return (attention_scores @ value), attention_scores
  
  def forward( self, q, k, v, mask=None ):
    # q,k,v: (batch, path_len, d_model)
    query = self.w_q(q)
    key = self.w_k(k)
    value = self.w_v(v)

    # q,k,v: (path_len, d_model) --> (path_len, num_heads, d_k) --> (num_heads, path_len, d_k)
    query = query.view(query.shape[0], self.num_heads, self.d_k).transpose(0, 1)
    key = key.view(key.shape[0], self.num_heads, self.d_k).transpose(0, 1)
    value = value.view( value.shape[0], self.num_heads, self.d_k).transpose(0, 1)

    # Calculate attentions for each head
    # output: ( num_heads, path_len, d_k)
    output, self.attention_scores = MultiHeadAttentionBlock.calcAttention(query, key, value, mask, self.dropout)

    # output: (num_heads, path_len, d_k) --> (path_len, num_heads, d_k)
    output = output.transpose(0, 1).contiguous()

    # output: (path_len, num_heads, d_k) --> (path_len, d_model)
    output = output.view(output.shape[0], self.num_heads * self.d_k)

    return self.w_o(output)
  
  
# ! layer = sublayer
# ResidualConnector - implements skip connection(with dropout) to keep track of progress and applies normalization layer
class ResidualConnector(nn.Module):
  def __init__( self, dropout: float ):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = Normalizator()
  
  def forward( self, input, layer ):

    # Apply normalization and Calculate output from the layer which will be skipped
    layer_output = layer( self.norm(input) )

    # Add the skipped layer to the initial input
    # ( Dropout is needed to avoid overfitting and getting stuck at local minima )
    return input + self.dropout(layer_output)


# Single Encoder block (Encoder can have up to 10 of them)
class EncoderBlock(nn.Module):
  def __init__( self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float ):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block = feed_forward_block
    # nn.ModuleList = pyTorch way of initializing an array of nn.Module instances
    self.resid_cons = nn.ModuleList([ResidualConnector(dropout), ResidualConnector(dropout)])
  
  def forward( self, input, src_mask):
    # Self Attentions
    output = self.resid_cons[0](input, lambda input: self.self_attention_block(input, input, input, src_mask))

    # Feed Forward
    return self.resid_cons[1](output, self.feed_forward_block)
  

# -- Encoder --
class Encoder(nn.Module):
  def __init__( self, layers: nn.ModuleList):
    super().__init__()
    self.layers = layers
    self.norm = Normalizator()
  
  def forward( self, input, mask ):
    current_output = input

    # Sequentially send input through every given EncoderBlock with the same mask
    for layer in self.layers:
      current_output = layer(current_output, mask)
    
    # Normalize final output
    return self.norm(current_output)


# Single Decoder block
class DecoderBlock(nn.Module):
  def __init__( self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float ):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.resid_cons = nn.ModuleList([ResidualConnector(dropout), ResidualConnector(dropout), ResidualConnector(dropout)])
  
  def forward( self, tgt_input, encoder_output, src_mask, tgt_mask):
    # Self Attentions for the tgt_input; with target mask
    tgt_output = self.resid_cons[0](tgt_input, lambda tgt_input: self.self_attention_block( tgt_input, tgt_input, tgt_input, tgt_mask ))

    # Cross Attentions for values, keys as Encoder output and queries as the decoder block output; with source mask
    tgt_output = self.resid_cons[1](tgt_output, lambda tgt_output: self.cross_attention_block( tgt_output, encoder_output, encoder_output, src_mask ))

    # Feed Forward
    return self.resid_cons[2](tgt_output, self.feed_forward_block)


# -- Decoder --
class Decoder(nn.Module):
  def __init__( self, layers: nn.ModuleList):
    super().__init__()
    self.layers = layers
    self.norm = Normalizator()
  
  def forward( self, tgt_input, encoder_output, src_mask, tgt_mask ):
    current_tgt_output = tgt_input

    # Sequentially send input through every given DecoderBlock with the same mask
    for layer in self.layers:
      current_tgt_output = layer(current_tgt_output, encoder_output, src_mask, tgt_mask)
    
    # Normalize final output
    return self.norm(current_tgt_output)


# ! num_nodes = vocab_size
# -- Projection layer: projects embeddings into the nodes --
# TODO: consider setting num_nodes to 4 as we have only 4 options where to go after each step
class ProjectionLayer(nn.Module):
  def __init__( self, d_model: int, num_nodes: int ):
    super().__init__()
    self.proj = nn.Linear(d_model, num_nodes)
  
  def forward( self, input ):
    # input: (batch, path_len, d_model) --> (batch, path_len, num_nodes)
    return self.proj(input)


# -- Transformer --
class Transformer(nn.Module):
  def __init__( self, gcn: GCNs, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedder, tgt_embedding: InputEmbedder, src_pos: PoistionalEncoder, tgt_pos: PoistionalEncoder, projection_layer: ProjectionLayer ):
    super().__init__()
    torch.manual_seed(1234567) # TODO: delete after dev phase is done
    self.gcn = gcn
    self.encoder = encoder
    self.decoder = decoder
    self.src_embedding = src_embedding
    self.tgt_embedding = tgt_embedding
    self.src_pos = src_pos
    self.tgt_pos = tgt_pos
    self.projection_layer = projection_layer
  
  def encode( self, src_input, edge_index, src_mask ):
    # GCN2 applied
    out = self.gcn(src_input, edge_index)

    # Calculating embeddings for GCN output (encoder input); then adding them to positional encodings 
    out = self.src_embedding(out)
    out = self.src_pos(out)

    # Ecnoder forward pass
    return self.encoder(out, src_mask)

  def decode( self, encoder_output, src_mask, tgt_input, max_path_len, training_mode ):

    # Add EOS to the beginning of the sequence
    eos = max_path_len-1
    if(training_mode == True):
      tgt_input = torch.cat((torch.tensor([eos]), tgt_input))
    else:
      tgt_input = torch.tensor([eos])

    tmp_mask = None
    finalOut = torch.tensor([])

    print(tgt_input, tgt_input.shape)

    # Loop
    for step in range(1, max_path_len):
      print("mask size: ", step)

      # Calculating embeddings for predicted path(decoder input); then adding them to positional encodings
      # TODO: consider applying masks to embedding layer as well
      out = self.tgt_embedding(tgt_input)
      out = self.tgt_pos(out)

      # Generate a mask, to be deleted after training is done
      if(training_mode == True):
        tmp_mask = torch.zeros((len(tgt_input), len(tgt_input)))
        tmp_mask[:step, :step] = 1

      # Decoder forward pass
      out = self.decoder(out, encoder_output, src_mask, tmp_mask)
      out = self.project(out)
      nextNode = torch.argmax(out[step-1])

      # Append next node to the final list of steps predicted
      finalOut = torch.cat((finalOut, out[step-1].unsqueeze(0)))

      if(training_mode == True):
        # Teacher Forcing, to be deleted after training
        if(step == len(tgt_input) - 1):
          print(finalOut, finalOut.shape)
          return finalOut
      else:
        # Update decoder input
        tgt_input = torch.cat((tgt_input, nextNode.unsqueeze(0)))
        
      
      # If EOS is reached => end of sequence => end the loop.
      if( nextNode == eos and training_mode == False ):
        return finalOut

    return finalOut
  
  def project( self, input ):
    # Final linear NN
    return self.projection_layer(input)

  def forward( self, encoder_input, decoder_input, adj_input, encoder_mask, max_path_len, training_mode=False ):
    encoder_output = self.encode( encoder_input, adj_input, encoder_mask )
    decoder_output = self.decode( encoder_output, encoder_mask, decoder_input, max_path_len, training_mode )
    return decoder_output


# -- Function to build a Transformer --
def transformer_builder( src_num_nodes: int, tgt_num_nodes: int, max_src_len: int, max_tgt_len: int, d_model: int=512, num_encoderBlocks: int=6, num_attnHeads: int=8, dropout: float=0.1, d_ff: int=2048, resume: bool=False  ) -> Transformer:
  # GCN layer
  gcn = GCNs(dropout)

  # Embedding layers
  src_embedding = InputEmbedder(d_model, src_num_nodes)
  tgt_embedding = InputEmbedder(d_model, tgt_num_nodes)

  # Positional Encoding layers
  src_posEnc = PoistionalEncoder(d_model, max_src_len, dropout)
  tgt_posEnc = PoistionalEncoder(d_model, max_tgt_len, dropout)

  # Encoder
  encoder_blocks = nn.ModuleList()

  for _ in range(num_encoderBlocks):
    self_attention = MultiHeadAttentionBlock(d_model, num_attnHeads, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    encoder_blocks.append( EncoderBlock(self_attention, feed_forward_block, dropout) )
  
  encoder = Encoder(encoder_blocks)

  # Decoder
  decoder_blocks = nn.ModuleList()

  for _ in range(num_encoderBlocks):
    self_attention = MultiHeadAttentionBlock(d_model, num_attnHeads, dropout)
    cross_attention = MultiHeadAttentionBlock(d_model, num_attnHeads, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    decoder_blocks.append( DecoderBlock(self_attention, cross_attention, feed_forward_block, dropout) )
  
  decoder = Decoder(decoder_blocks)

  # Projection Layer
  projection_layer = ProjectionLayer(d_model, max_tgt_len)

  # Transformer
  transformer = Transformer(gcn, encoder, decoder, src_embedding, tgt_embedding, src_posEnc, tgt_posEnc, projection_layer)

  # Resume parameters if continuing previous training
  if(resume):
    checkpoint = torch.load('./savedGrads/checkpoint.pth.tar')
    transformer.load_state_dict(checkpoint['model_state_dict'])

    return checkpoint, transformer

  # Init parameters if starting new training
  else:
    for p in transformer.parameters():
      if p.dim() > 1:
          nn.init.xavier_uniform_(p)

    return None, transformer