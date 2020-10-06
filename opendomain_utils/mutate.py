import numpy as np
import copy
import random

def is_full_dag(matrix):
  """Full DAG == all vertices on a path from vert 0 to (V-1).

  i.e. no disconnected or "hanging" vertices.

  It is sufficient to check for:
    1) no rows of 0 except for row V-1 (only output vertex has no out-edges)
    2) no cols of 0 except for col 0 (only input vertex has no in-edges)

  Args:
    matrix: V x V upper-triangular adjacency matrix

  Returns:
    True if the there are no dangling vertices.
  """
  shape = np.shape(matrix)

  rows = matrix[:shape[0]-1, :] == 0
  rows = np.all(rows, axis=1)     # Any row with all 0 will be True
  rows_bad = np.any(rows)

  cols = matrix[:, 1:] == 0
  cols = np.all(cols, axis=0)     # Any col with all 0 will be True
  cols_bad = np.any(cols)

  return (not rows_bad) and (not cols_bad)

def mutate_adj(old_adj):
    new_adj = copy.deepcopy(old_adj)
    node_to_connect = random.choice(list(range(0,5)))
    possible_connects = 6- node_to_connect
    to_connect = random.choice(list(range(possible_connects))) + node_to_connect + 1
    node_to_disconnect = random.choice(list(range(0, 5)))
    possible_disconnects = [i for i in range(node_to_disconnect+1,7) if new_adj[node_to_disconnect][i] == 1]
    to_disconnect = random.choice(possible_disconnects)
    new_adj[node_to_connect, to_connect] = 1
    new_adj[node_to_disconnect, to_disconnect] = 0
    # print("node to connect:{}, to connect:{}, node to disconnect:{}, to disconnect:{}".format(node_to_connect,to_connect,node_to_disconnect,to_disconnect))
    return new_adj

def mutate_ops(old_ops):
    new_ops = copy.deepcopy(old_ops)
    node_to_mutate = random.choice(list(range(1,6)))
    op_to_set = random.choice(list(range(0,3)))
    new_ops[node_to_mutate] = 0
    new_ops[node_to_mutate][op_to_set] = 1
    return new_ops

def mutate_arch(old_arch):
    change = random.choice([0,1,2])
    if change==0:
        new_arch = {'adjacency_matrix': mutate_adj(old_arch['adjacency_matrix']), "operations": old_arch['operations']}
    elif change==1:
        new_arch = {'adjacency_matrix': old_arch['adjacency_matrix'], "operations": mutate_ops(old_arch['operations'])}
    else:
        new_arch = {'adjacency_matrix': mutate_adj(old_arch['adjacency_matrix']), "operations": mutate_ops(old_arch['operations'])}

    return new_arch

def all_mutates(old_arch):
    new_archs = []
    strs = []
    for i in range(30000):
        mutated = mutate_arch(old_arch)
        if is_full_dag(mutated['adjacency_matrix']) and str(mutated) not in strs:
            strs.append(str(mutated))
            new_archs.append(mutated)
    return new_archs
