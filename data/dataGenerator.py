import random
import pandas as pd
import torch
import torch_geometric.utils as tg
from collections import deque
import sys
import math

# Algorithm for finding the optimal path
def bfs(adjacency_matrix, S, par, dist):
  # Preparing graph by convertin long tesnor into int list 
  graph = adjacency_matrix

  # Queue to store the nodes in the order they are visited
  q = deque()
  # Mark the distance of the source node as 0
  dist[S] = 0
  # Push the source node to the queue
  q.append(S)

  # Iterate until the queue is not empty
  while q:
      # Pop the node at the front of the queue
      node = q.popleft()

      # Explore all the neighbors of the current node
      for neighbor in graph[node]:
          # Check if the neighboring node is not visited
          if dist[neighbor] == float('inf'):
              # Mark the current node as the parent of the neighboring node
              par[neighbor] = node
              # Mark the distance of the neighboring node as the distance of the current node + 1
              dist[neighbor] = dist[node] + 1
              # Insert the neighboring node to the queue
              q.append(neighbor)
    
  return dist


def get_shortest_distance(adjacency_matrix, S, D, V):
  # par[] array stores the parent of nodes
  par = [-1] * V

  # dist[] array stores the distance of nodes from S
  dist = [float('inf')] * V

  # Function call to find the distance of all nodes and their parent nodes
  dist = bfs(adjacency_matrix, S, par, dist)

  if dist[D] == float('inf'):
      print("Source and Destination are not connected")
      return

  # List path stores the shortest path
  path = []
  current_node = D
  path.append(D)
  while par[current_node] != -1:
      path.append(par[current_node])
      current_node = par[current_node]

  # Printing path from source to destination
  return path

def generate_dataset( num_nodes, imperfect=False):
    if(math.sqrt(num_nodes) % 1):
        sys.exit(f"Number of nodes = {num_nodes} can't form a grid layout")

    # TODO: no repetitions
    # Max number of samples = num_nodes^2

    dataset = []
    n_imperfect_samples = 0

    # Dynamically generating edge_index
    edge_index = [[], []]
    n_rows = int(math.sqrt(num_nodes)) # n_rows = num of columns = number of elems per row

    for row in range(n_rows):
        for elem in range(n_rows):
            current_elem = elem + row*n_rows
            # lower neighbor
            if( current_elem + n_rows < num_nodes ):
                edge_index[0].append(current_elem)
                edge_index[1].append(current_elem+n_rows)
                edge_index[0].append(current_elem+n_rows)
                edge_index[1].append(current_elem)
            # right neighbor
            if( elem + 1 < n_rows ):
                edge_index[0].append(current_elem)
                edge_index[1].append(current_elem+1)
                edge_index[0].append(current_elem+1)
                edge_index[1].append(current_elem)
            # left neighbor
            if( elem - 1 > 0 ):
                edge_index[0].append(current_elem)
                edge_index[1].append(current_elem-1)
                edge_index[0].append(current_elem-1)
                edge_index[1].append(current_elem)
            # upper neighbor
            if( current_elem - n_rows > 0 ):
                edge_index[0].append(current_elem)
                edge_index[1].append(current_elem-n_rows)
                edge_index[0].append(current_elem-n_rows)
                edge_index[1].append(current_elem)

    # List to store the graph as an adjacency list
    graph = [[] for _ in range(num_nodes)]

    # Generate graph based on edge_index
    for i, node in enumerate(edge_index[0]):
        graph[node].append(edge_index[1][i])

    # Generate samples with all the possible source and destination nodes
    for row_id in range(num_nodes):
        for column_id in range(num_nodes):
            # Generating random source and destination nodes
            source = row_id
            destination = column_id
            
            if( imperfect ):
                # Find optimal path and sometimes longer path
                if( random.random() < 0.05):
                    # Randomly longer path
                    in_between_node = random.randint(0, num_nodes-1)
                    path = get_shortest_distance(graph, source, in_between_node, num_nodes)[::-1]
                    path.extend(get_shortest_distance(graph, in_between_node, destination, num_nodes)[::-1][1:])

                    # Y = nodes to go to to reach destination. Minimum size: 1
                    # path is reversed to follow s->d route
                    Y = [path,[1]]
                    n_imperfect_samples += 1
                else:
                    # Optimal path
                    path = get_shortest_distance(graph, source, destination, num_nodes)

                    # Y = nodes to go to to reach destination. Minimum size: 1
                    # path is reversed to follow s->d route
                    Y = [path[::-1],[0]]

            else:
                # Find optimal path
                path = get_shortest_distance(graph, source, destination, num_nodes)

                # Y = nodes to go to to reach destination. Minimum size: 1
                # path is reversed to follow s->d route
                Y = [path[::-1],[0]]
            
            # X = nx1 size list of values of each node
            X = [[0]] * num_nodes
            X[source] = [5]
            X[destination] = [10]
            
            dataset.append([edge_index, X, Y])

    return dataset, n_imperfect_samples


# Main params - Generate a dataset -----------------------------------------------------------------
"""
num_nodes - number of nodes in a grid
imperfect - bool to make a dataset full of either mixed or perfect samples
"""
# If changing imperfect_dataset - Make sure "raw_file_names()" returns correct file in dataset.py
imperfect_dataset = False
dataset, n_imperfect_samples = generate_dataset(num_nodes=100, imperfect=imperfect_dataset)

# Create a DataFrame
df = pd.DataFrame(dataset, columns=["Edge index", "X", "Y"])


if( imperfect_dataset == True):
    # Write the DataFrame to an CSV file
    df.to_csv("./raw/imperfect.csv", index=False)
else:
    # Write the DataFrame to an CSV file
    df.to_csv("./raw/perfect.csv", index=False)

print(f"Number of samples: {len(dataset)}")
print(f"Number of imperfect samples: {n_imperfect_samples}")