import torch
import torch.nn as nn
from models.GTN2 import transformer_builder
from torch.optim.lr_scheduler import LambdaLR

from visualization import visualizeGraph, visualizeLoss
from data.dataset import PredictShortestPathDataset
from functions import get_data_subset, get_eval_subset, save_checkpoint, is_correct, generate_enc_mas, validate_curr_epoch


# TODO: try deeper model for 15 epochs
# TODO: if doesn't work try 80% dataset for 10x10 grids


import sys # TODO: delete after done with development

# TODO: Compare results of batch_size = 20 and = 100. Try different hyperparameters !
# TODO: Try to retrain the model. Record outcome

# TODO: Post-training Q/A refining(find incorrect answers and use loss function with optimizer to nodge the weights)
# TODO: Try to eval the model. Record outcome

# TODO: MAYBE: Training dataset is too small and doesn’t cover all the optimal paths
# TODO: Try to run the model. Record outcome

# TODO: MAYBE: For the linear part after decoder before softmax, change dimensions of linear NN to 4.
# TODO: Try to run the model. Record outcome

# -- Device --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Chosen device for training: {device}')


# -- Config --
config = {
  "batch_size": 50,
  "num_epochs": 30,
  # TODO: start with bigger one and gradually go down to 10**-4 or some other small number
  "lr": 10**-4,
  "num_nodes": 2500,
  "num_samples": 25000
}


# -- Dataset --
dataset = PredictShortestPathDataset(root="./data")


# -- Visualize a single data sample --
visualizeGraph(dataset, num_nodes=2500, run=False)


# -- Model & Optimizer & Criterion --
# Modifications 10/24/24: d_model=774(was 512), num_encoderBlocks=7(was 6), num_attnHeads=9(was 8), d_ff=3072(was 2048), epoch size = 25000(was 12500)
checkpoint, model = transformer_builder( max_src_len=config['num_nodes']+1, max_tgt_len=config['num_nodes']+1, d_model=512, num_encoderBlocks=6, num_attnHeads=8, dropout=0.1, d_ff=2048, resume=False, device=device )
model.to(device)

currTimeStep = 1
Twarmup = 15 # warmup will take place during the first 10 mini-batches
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']*currTimeStep/Twarmup, eps=1e-9)

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

# Validation / Evaluation variables
validLoss = []

# Getting universal edge index
edge_index_sample = next(iter(get_data_subset(dataset, batch_size=1, n_samples=1)))[0].edge_index.to(device)
edge_set_sample = set(zip(edge_index_sample[0].tolist(), edge_index_sample[1].tolist()))

# Encoder mask
encoder_mask = generate_enc_mas(num_nodes=config['num_nodes'], edge_set=edge_set_sample).to(device)

for epoch in range(config['num_epochs']):
  # One epoch
  trainLoader = get_data_subset(dataset, batch_size=config['batch_size'], n_samples=config['num_samples'])

  for batch_index, batch in enumerate(trainLoader):
    # One batch
    optimizer.zero_grad()
    temp_losses = list()
    
    for i in range(config['batch_size']):
      # One sample
      
      # X
      encoder_input = batch[i].x.to(device)
      # Edge Index list
      adj_input = edge_index_sample
      
      # y, to be deleted after the model is trained
      decoder_input = batch[i].y.to(device)
      # y_flag (represents whether a sample is not an optimal path), to be deleted after the model is trained
      y_flag = batch[i].imperfect_y_flag.item()

      # Generate prediction (we're interested in probs, not the steps)
      _, prediction = model( encoder_input, decoder_input, adj_input, encoder_mask, config['num_nodes']+1, training_mode=True )

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

    # Update lr (warmup)
    if( currTimeStep < Twarmup ):
      currTimeStep += 1
      for g in optimizer.param_groups:
        g['lr'] = config['lr']*currTimeStep/Twarmup
    
    # Save average loss of the batch
    avg_batch_loss = (sum(temp_losses) / len(temp_losses))
    losses.append(avg_batch_loss)

    print(f"Epoch: {epoch+1}, Batch: {batch_index}, Loss: {avg_batch_loss}")
  
  # Validate current epoch
  del trainLoader 
  
  #validIter = iter( get_data_subset(dataset, batch_size=250, n_samples=config['num_samples']) )
  #validLoss.append(validate_curr_epoch( validIter, edge_index_sample, edge_set_sample, model, encoder_mask, config, device ))
  #del validIter


# -- Save progress of training --
save_checkpoint({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_epochs': config['num_epochs'],
            'prevConfig': config
            })
print(f'The model has been saved at {config["num_epochs"]} epochs')


# -- Visualization of loss curve --
visualizeLoss([losses, validLoss], run=True)


# -- Evaluation --
model.eval()
evalIter = iter( get_eval_subset(dataset, valid_percantage=0.005) )

with torch.no_grad():
  complete_success_rate = []

  while True:
    try:
      # Get the next validation / evaluation batch
      batch = next(evalIter)

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

        # print("Current prediction reviewed: ", prediction)
        # print("Correct answer: ", label)
        
        # Check if all the nodes are correct and src and dest are correct
        complete_success_rate.append( is_correct(encoder_input, edge_set_sample, prediction) )
      
      print("100 more samples have been Evaluated...")
    except StopIteration:
      # Avoid raising StopIteration
      break

print(f"Complete success percentage (length and all elements are correct): {(sum(complete_success_rate) / len(complete_success_rate)) * 100 }%")
