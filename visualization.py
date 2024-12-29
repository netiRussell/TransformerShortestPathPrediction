from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

def visualizeGraph(dataset, num_nodes, run):
  if run == False:
    return

  # Dataset info
  data = dataset[0]

  # Visualization
  G = to_networkx(data, to_undirected=True)
  plt.figure(figsize=(7,7))
  plt.xticks([])
  plt.yticks([])
  nx.draw_networkx(G,
                  pos=nx.bfs_layout(G, 0),
                  with_labels=True)
  plt.show()


def visualizeLoss(losses, run):
  if run == False:
    return
  
  fig, (plt1, plt2) = plt.subplots(1,2)
  fig.suptitle("Losses")

  plt1.plot(range(1, len(losses[0]) + 1), losses[0], marker='o')
  plt1.set_title('Training Loss Curve')
  plt1.set_xlabel('Batch')
  plt1.set_ylabel('Average loss')
  plt1.grid()

  plt2.plot(range(1, len(losses[1]) + 1), losses[1], marker='o')
  plt2.set_title('Validation per epoch')
  plt2.set_xlabel('Epoch')
  plt2.set_ylabel('Percantage')
  plt2.grid()

  plt.show()