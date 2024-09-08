import torch
import torch.nn as nn
from models.GTN2 import transformer_builder

from visualization import visualizeGraph, visualizeLoss
from data.dataset import PredictShortestPathDataset
from functions import prepare_data, save_checkpoint, is_correct


import sys # TODO: delete after done with development

# TODO: Take X out, so that its computed and prepared only once.
# TODO: Apply masks(no self loop, only connections with neighbors) in encoder based on the adjacency matrix to avoid jumping from nodes that are not connected
# TODO: Try to run the model. Record outcome

# TODO: Implement dynamic learning rate
# TODO: Try to run the model. Record outcome

# TODO: Training dataset is too small and doesn’t cover all the optimal paths
# TODO: Try to run the model. Record outcome

# TODO: Compare results of batch_size = 20 and = 100. Try different hyperparameters
# TODO: Try to run the model. Record outcome

# TODO: For the linear part after decoder before softmax, change dimensions of linear NN to 4.
# TODO: Try to run the model. Record outcome

# -- Device --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Chosen device for training: {device}')


# -- Config --
config = {
  "batch_size": 50,
  "num_epochs": 8,
  # TODO: start with bigger one and gradually go down to 10**-4 or some other small number
  "lr": 10**-4,
  "num_nodes": 100
}


# -- Dataset --
dataset = PredictShortestPathDataset(root="./data")
total_samples = len(dataset)

trainLoader, validLoader = prepare_data( dataset=dataset, batch_size=config['batch_size'], valid_percantage=0.1)


# -- Visualize a single data sample --
visualizeGraph(dataset, num_nodes=100, run=False)


# -- Model & Optimizer & Criterion --

# max_src_len - max path length for source including EOS to start and end with
# max_tgt_len - max path length for tgt including EOS to start with
checkpoint, model = transformer_builder( src_num_nodes=config['num_nodes'], tgt_num_nodes=config['num_nodes'], max_src_len=config['num_nodes']+1, max_tgt_len=config['num_nodes']+1, d_model=512, num_encoderBlocks=6, num_attnHeads=8, dropout=0.1, d_ff=2048, resume=False )
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

# TODO: make sure loss doesn't count EOS by utilizing ignore_index param
criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)


# -- Load model & optimizer --
if ( False ):
  checkpoint = torch.load('./savedGrads/checkpoint.pth.tar')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  total_epochs = checkpoint['total_epochs']
  prevConfig = checkpoint['prevConfig']

  print(f'Training is resumed! Starting from epoch #{total_epochs}')


# -- Training Loop --
model.train()
losses = list()

# mask relations between nodes that are not neighbors
encoder_mask = None # (1, Seq_Len)

for epoch in range(config['num_epochs']):
  # One epoch
  for batch_index, batch in enumerate(trainLoader):
    # One batch
    optimizer.zero_grad()
    temp_losses = list()
    
    for i in range(config['batch_size']):
      # One sample
      
      # X
      encoder_input = batch[i].x.to(device)
      # Edge Index list
      adj_input = batch[i].edge_index.to(device)
      
      # TODO: undesrtand what is decoder_input in the source code
      # y, to be deleted after the model is trained
      decoder_input = batch[i].y.to(device)
      # y_flag (represents whether a sample is not an optimal path), to be deleted after the model is trained
      y_flag = batch[i].imperfect_y_flag.item()

      # Generate prediction (we're interested in probs, not the steps)
      _, prediction = model( encoder_input, decoder_input, adj_input, encoder_mask, config['num_nodes']+1, device, training_mode=True )

      # Add EOS to the end of the current label(y)
      decoder_input = torch.cat( (decoder_input, torch.tensor([config['num_nodes']]).to(device)) )
      
      # Loss, to be deleted after the model is trained
      loss = criterion(prediction.contiguous(), decoder_input.contiguous()).to(device)

      # Save loss
      temp_losses.append(loss.item())

      # Imperfect sample case
      if(y_flag == 1):
        loss = 0.15*loss
      
      # Backpropagation
      loss.backward()
    
    # Update weights after every batch
    optimizer.step()
    
    # Save average loss of the batch
    avg_batch_loss = (sum(temp_losses) / len(temp_losses))
    losses.append(avg_batch_loss)

    print(f"Epoch: {epoch+1}, Batch: {batch_index}, Loss: {avg_batch_loss}")


# -- Save progress of training --
if('total_epochs' in locals()):
  config['num_epochs'] += total_epochs

num_epochs = config['num_epochs']

save_checkpoint({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_epochs': config['num_epochs'],
            'prevConfig': config
            })
print(f'The model has been saved at {num_epochs} epochs')


# -- Visualization of loss curve --
visualizeLoss(losses, run=True)


# -- Evaluation --
model.eval()

with torch.no_grad():
  complete_success_rate = []

  for batch_index, batch in enumerate(validLoader):
    for i in range(config['batch_size']):
      # Imperfect sample; to be disregarded
      if( y_flag == 1 ):
        continue
      
      # X
      encoder_input = batch[i].x.to(device)
      # Edge Index list
      adj_input = batch[i].edge_index.to(device)
      # y
      label = batch[i].y.to(device)

      # Generate prediction (we're interested in steps, not the probs)
      prediction, _ = model( encoder_input, None, adj_input, encoder_mask, config['num_nodes']+1, device)

      # Check if the length of the output is correct
      if(len(label) != len(prediction)):
        complete_success_rate.append(0)
        continue

      print("Current prediction reviewed: ", prediction)
      
      # TODO: no need to redo the same edge_list
      # Check if all the nodes are correct and src and dest are correct
      complete_success_rate.append( is_correct(encoder_input, adj_input, prediction) )
    
    print(f"Evaluation is in the process... Current batch = {batch_index}")

  print(f"Complete success percentage (length and all elements are correct): {(sum(complete_success_rate) / len(complete_success_rate)) * 100 }%")
