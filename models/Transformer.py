import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import math
import sys # TODO: delete after dev phase is done

class CustomSigmoid(nn.Module):
  def __init__(self):
    super(CustomSigmoid, self).__init__()

  def forward(self, x, beta=1):
    return len(x)*torch.sigmoid(x)
  

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    # Ensure that the model dimension (d_model) is divisible by the number of heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    # Initialize dimensions
    self.d_model = d_model # Model's dimension
    self.num_heads = num_heads # Number of attention heads
    self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
    
    # Linear layers for transforming inputs
    self.W_q = nn.Linear(d_model, d_model) # Query transformation
    self.W_k = nn.Linear(d_model, d_model) # Key transformation
    self.W_v = nn.Linear(d_model, d_model) # Value transformation
    self.W_o = nn.Linear(d_model, d_model) # Output transformation
      
  def scaled_dot_product_attention(self, Q, K, V, mask=None):
    # Calculate attention scores
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    
    # Apply mask if provided (useful for preventing attention to certain parts like padding)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    
    # Softmax is applied to obtain attention probabilities
    attn_probs = torch.softmax(attn_scores, dim=-1)
    
    # Multiply by values to obtain the final output
    output = torch.matmul(attn_probs, V)
    return output
      
  def split_heads(self, x):
    # Reshape the input to have num_heads for multi-head attention
    batch_size, seq_length, d_model = x.size()
    return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
      
  def combine_heads(self, x):
    # Combine the multiple heads back to original shape
    batch_size, _, seq_length, d_k = x.size()
    return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
      
  def forward(self, Q, K, V, mask=None):
    # Apply linear transformations and split heads
    Q = self.split_heads(self.W_q(Q))
    K = self.split_heads(self.W_k(K))
    V = self.split_heads(self.W_v(V))
    
    # Perform scaled dot-product attention
    attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
    
    # Combine heads and apply output transformation
    output = self.W_o(self.combine_heads(attn_output))
    return output


# NN for self-attention precision
class PositionWiseFeedForward(nn.Module):
  def __init__(self, d_model, d_ff):
    super(PositionWiseFeedForward, self).__init__()
    self.fc1 = nn.Linear(d_model, d_ff)
    self.fc2 = nn.Linear(d_ff, d_model)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    return self.fc2(out)


# Helps to keep track of each element in the sequence
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length):
    super(PositionalEncoding, self).__init__()
    
    pe = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    self.register_buffer('pe', pe.unsqueeze(0))
      
  def forward(self, x):
    return x + self.pe[:, :x.size(1)]


# Encoder
class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super(EncoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, x, mask):
    attn_output = self.self_attn(x, x, x, mask)
    x = self.norm1(x + self.dropout(attn_output))
    ff_output = self.feed_forward(x)
    x = self.norm2(x + self.dropout(ff_output))
    return x


# Decoder
class DecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super(DecoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.cross_attn = MultiHeadAttention(d_model, num_heads)
    self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, x, enc_output, src_mask, tgt_mask):
    attn_output = self.self_attn(x, x, x, tgt_mask) # TODO: maybe mask is not needed since I have explicitly implemented one
    x = self.norm1(x + self.dropout(attn_output))
    attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
    x = self.norm2(x + self.dropout(attn_output))
    ff_output = self.feed_forward(x)
    x = self.norm3(x + self.dropout(ff_output))
    return x


# Transformer
class Transformer(nn.Module):
  def __init__(self, src_size, target_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
    super(Transformer, self).__init__()
    torch.manual_seed(1234567) # TODO: delete after dev phase is done

    self.max_seq_length = max_seq_length

    # ! Should I have a gradual increase and decrease of node numbers?
    # num GCN layers = sqrt(n_nodes) - 1
    self.gcn1 = GCNConv(1, target_size)
    self.gcn2 = GCNConv(target_size, target_size)
    self.gcn3 = GCNConv(target_size, target_size)
    self.gcn4 = GCNConv(target_size, target_size)
    self.gcn5 = GCNConv(target_size, target_size)
    self.gcn6 = GCNConv(target_size, target_size)
    self.gcn7 = GCNConv(target_size, int(target_size/2))
    self.gcn8 = GCNConv(int(target_size/2), int(target_size/4))
    self.gcn9 = GCNConv(int(target_size/4), 1)
    self.sigmoidNumNod = CustomSigmoid()

    self.encoder_embedding = nn.Embedding(src_size, d_model)
    self.decoder_embedding = nn.Embedding(target_size, d_model)
    self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

    self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    self.fc = nn.Linear(d_model, target_size)
    self.dropout = nn.Dropout(dropout)

  def generate_mask_src(self, src):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    seq_length = src.size(1)
    return src_mask
  
  def generate_mask_tgt(self, tgt):
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return tgt_mask

  def forward(self, src, y, adj, train_status):
    # -- GCN --
    out = self.gcn1(src[0].unsqueeze(-1).float(), adj)
    out = self.dropout(out)

    out = F.leaky_relu(self.gcn2(out, adj))
    out = self.dropout(out)

    out = F.relu(self.gcn3(out, adj))
    out = self.dropout(out)

    out = F.relu(self.gcn4(out, adj))
    out = self.dropout(out)

    out = torch.sigmoid((self.gcn5(out, adj)))
    out = self.dropout(out)

    out = F.relu(self.gcn6(out, adj))
    out = self.dropout(out)

    out = F.relu(self.gcn7(out, adj))
    out = self.dropout(out)

    out = F.relu(self.gcn8(out, adj))
    out = self.dropout(out)

    out = self.sigmoidNumNod(self.gcn9(out, adj))
    out = out.squeeze(1).unsqueeze(0).long()

    # -- Encoder --
    src_mask = self.generate_mask_src(out)
    src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(out)))

    enc_output = src_embedded
    for enc_layer in self.encoder_layers:
        enc_output = enc_layer(enc_output, src_mask)

    # -- Decoder --
    # Initial input for Encoder = EOS = last node id + 1
    num_nodes = len(out[0])
    eos_token = num_nodes
    tgt = torch.tensor([[eos_token]])
    final_prediction = torch.empty(0)

    for i in range(self.max_seq_length-1):
      # Masks and Positional embeddings calcualted
      tgt_mask = self.generate_mask_tgt(tgt)
      tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

      # Input is passed through the decoder
      dec_output = tgt_embedded
      for dec_layer in self.decoder_layers:
          dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
      output = self.fc(dec_output)

      # Find next step based on the prediction of the last element
      next_step = torch.argmax(output[0][-1]).unsqueeze(0)

      # Append last element of output to save prediction
      final_prediction = torch.cat([final_prediction, output[0][-1].view(1, num_nodes+1)])
      
      # EOS is reached => sequence has ended
      # TODO: delete "and train_status == False" after dev phase is done
      if ( next_step.item() == eos_token and train_status == False ):
        return final_prediction
      
      # - Teacher Forcing. For training only -
      # TODO: delete Teacher Forcing after dev phase is done
      if (train_status == True ):
        # If transformer wants to continue path prediction even though EOS is reached in the label(y) => force it to stop
        # There is no need to use last elem from labels in tgt since that elem is an EOS and no prediction of what will go after EOS is needed because nothing is supposed to go after EOS.
        if( i == len(y[0]) - 1 ):
          return final_prediction

        # New decoder input defined with a correct next step
        tgt = torch.cat((tgt[0], y[0][i].view(1)), 0).unsqueeze(0)
      else:
        # New decoder input with a new step added to the current sequence
        tgt = torch.cat((tgt[0], next_step), 0).unsqueeze(0)

    return final_prediction