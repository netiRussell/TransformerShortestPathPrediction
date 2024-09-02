import random
import pandas as pd
import torch
import torch_geometric.utils as tg

from typing import List
from sys import maxsize, exit
from collections import deque
import math
import numpy
import sys #todo: delete

def find_paths(paths: List[List[int]], path: List[int], parent: List[List[int]], n: int, u: int) -> None:
    # Base Case
    if (u == -1):
        paths.append(path.copy())
        return
 
    # Loop for all the parents
    # of the given vertex
    for par in parent[u]:
 
        # Insert the current
        # vertex in path
        path.append(u)
 
        # Recursive call for its parent
        find_paths(paths, path, parent, n, par)
 
        # Remove the current vertex
        path.pop()
 
# Function which performs bfs
# from the given source vertex
def bfs(adj: List[List[int]], parent: List[List[int]], n: int,
        start: int) -> None:
 
    # dist will contain shortest distance
    # from start to every other vertex
    dist = [maxsize for _ in range(n)]
    q = deque()
 
    # Insert source vertex in queue and make
    # its parent -1 and distance 0
    q.append(start)
    parent[start] = [-1]
    dist[start] = 0
 
    # Until Queue is empty
    while q:
        u = q[0]
        q.popleft()

        for v in adj[u]:
            if (dist[v] > dist[u] + 1):

                # A shorter distance is found
                # So erase all the previous parents
                # and insert new parent u in parent[v]
                dist[v] = dist[u] + 1
                q.append(v)
                parent[v].clear()
                parent[v].append(u)

            elif (dist[v] == dist[u] + 1):

                # Another candidate parent for
                # shortes path found
                parent[v].append(u)
 
# Function which prints all the paths
# from start to end
def get_optimal_paths(adj: List[List[int]], n: int, start: int, end: int) -> None:
    paths = []
    parent = [[] for _ in range(n)]

    # Function call to bfs
    bfs(adj, parent, n, start)

    # Function call to find_paths
    find_paths(paths, [], parent, n, end)

    # Turn paths list into numpy array
    paths = numpy.array(paths)
    # Reverse the array and turn it back into a list
    paths = paths[:, ::-1].tolist()

    return paths

def add_edge(adj: List[List[int]],
             src: int, dest: int) -> None:
    adj[src].append(dest)
    adj[dest].append(src)


# Create a graph

adj = [[10, 1], [0, 11, 2], [1, 12, 3], [2, 13, 4], [3, 14, 5], [4, 15, 6], [5, 16, 7], [6, 17, 8], [7, 18, 9], [8, 19], [0, 20, 11], [1, 10, 21, 12], [2, 11, 22, 13], [3, 12, 23, 14], [4, 13, 24, 15], [5, 14, 25, 16], [6, 15, 26, 17], [7, 16, 27, 18], [8, 17, 28, 19], [9, 18, 29], [10, 30, 21], [11, 20, 31, 22], [12, 21, 32, 23], [13, 22, 33, 24], [14, 23, 34, 25], [15, 24, 35, 26], [16, 25, 36, 27], [17, 26, 37, 28], [18, 27, 38, 29], [19, 28, 39], [20, 40, 31], [21, 30, 41, 32], [22, 31, 42, 33], [23, 32, 43, 34], [24, 33, 44, 35], [25, 34, 45, 36], [26, 35, 46, 37], [27, 36, 47, 38], [28, 37, 48, 39], [29, 38, 49], [30, 50, 41], [31, 40, 51, 42], [32, 41, 52, 43], [33, 42, 53, 44], [34, 43, 54, 45], [35, 44, 55, 46], [36, 45, 56, 47], [37, 46, 57, 48], [38, 47, 58, 49], [39, 48, 59], [40, 60, 51], [41, 50, 61, 52], [42, 51, 62, 53], [43, 52, 63, 54], [44, 53, 64, 55], [45, 54, 65, 56], [46, 55, 66, 57], [47, 56, 67, 58], [48, 57, 68, 59], [49, 58, 69], [50, 70, 61], [51, 60, 71, 62], [52, 61, 72, 63], [53, 62, 73, 64], [54, 63, 74, 65], [55, 64, 75, 66], [56, 65, 76, 67], [57, 66, 77, 68], [58, 67, 78, 69], [59, 68, 79], [60, 80, 71], [61, 70, 81, 72], [62, 71, 82, 73], [63, 72, 83, 74], [64, 73, 84, 75], [65, 74, 85, 76], [66, 75, 86, 77], [67, 76, 87, 78], [68, 77, 88, 79], [69, 78, 89], [70, 90, 81], [71, 80, 91, 82], [72, 81, 92, 83], [73, 82, 93, 84], [74, 83, 94, 85], [75, 84, 95, 86], [76, 85, 96, 87], [77, 86, 97, 88], [78, 87, 98, 89], [79, 88, 99], [80, 91], [81, 90, 92], [82, 91, 93], [83, 92, 94], [84, 93, 95], [85, 94, 96], [86, 95, 97], [87, 96, 98], [88, 97, 99], [89, 98]]

# Given source and destination
src = 0
dest = 12

# Function Call
print(get_optimal_paths(adj, 100, src, dest))

sys.exit("still in development phase...")


# TODO: implement this logic in csv files generation
    # TODO: generate the regular perfect.csv where each sample is independent even if X is the same
    # TODO: generate eval.csv where all samples with the same X are grouped up in a single row
        # eval.csv will be used during evaluation to allow the model to choose optimal paths


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



# -----------------------------------------------------------------------------------
def generate_dataset( num_nodes, imperfect=False):
    if(math.sqrt(num_nodes) % 1):
        exit(f"Number of nodes = {num_nodes} can't form a grid layout")

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
        if( edge_index[1][i] not in graph[node] ):
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