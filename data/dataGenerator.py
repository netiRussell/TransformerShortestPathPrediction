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

            # X = nx1 size list of values of each node
            X = [[0]] * num_nodes
            X[source] = [5]
            X[destination] = [10]
            
            if( imperfect ):
                sys.exit("Is not adapted to multiple optimal paths") # Utilize dataset.append([edge_index, X, Y]) at the end
                # Find optimal path and sometimes longer path
                if( random.random() < 0.05):
                    # Randomly longer path
                    in_between_node = random.randint(0, num_nodes-1)
                    path = get_optimal_paths(graph, num_nodes, source, in_between_node,)[::-1]
                    path.extend(get_optimal_paths(graph, num_nodes, in_between_node, destination)[::-1][1:])

                    # Y = nodes to go to to reach destination. Minimum size: 1
                    # path is reversed to follow s->d route
                    Y = [path,[1]]
                    n_imperfect_samples += 1
                else:
                    # Optimal path
                    path = get_optimal_paths(graph, num_nodes, source, destination)

                    # Y = nodes to go to to reach destination. Minimum size: 1
                    # path is reversed to follow s->d route
                    Y = [path[::-1],[0]]

            else:
                # Find optimal path
                paths = get_optimal_paths(graph, num_nodes, source, destination)

                for path in paths:
                    # Y = nodes to go to to reach destination. Minimum size: 1
                    # path is reversed to follow s->d route
                    Y = [path,[0]]

                    # Add the path to the csv file
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