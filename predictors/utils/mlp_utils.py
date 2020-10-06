import numpy as np

def get_arch_acc(graph, maxsize=7):
    adjacency_matrix = np.array(graph['adjacency_matrix'], dtype=np.float32)
    adjacency_matrix = adjacency_matrix.flatten()
    while len(adjacency_matrix) < maxsize*maxsize:
        adjacency_matrix = np.append(adjacency_matrix, [0])
    operations = np.array(graph['operations'])
    operations = resnets_net_encoder(operations, maxsize=maxsize)
    architecture = np.concatenate((adjacency_matrix, operations), axis=None)
    architecture = np.array(architecture, dtype=np.float32)
    accuracy = np.array(graph['metrics'], dtype=np.float32)
    sample = {'architecture': architecture, 'accuracy': accuracy}
    return sample

def net_encoder(operations):
    operations = np.argmax(operations, 1)
    operations = np.array(operations, dtype = np.int32)
    for i in range(len(operations)):
        if operations[i] == 3: #input
            operations[i]=2
        elif operations[i] == 0: #conv1
            operations[i]=3
        elif operations[i] == 2: #pool
            operations[i]=4
        elif operations[i] == 1: #conv3
            operations[i]=5
        elif operations[i] == 4: #output
            operations[i]=6
    while len(operations) < 7:
        operations = np.append(operations, [9])
    return operations

def resnets_net_encoder(operations,maxsize=7):
    operations = np.argmax(operations, 1)
    operations = np.array(operations, dtype = np.int32)
    for i in range(len(operations)):
        if operations[i] == 0: #input
            operations[i]=1
        elif operations[i] == 1: #conv1
            operations[i]=2
        elif operations[i] == 2: #pool
            operations[i]=3
    while len(operations) < maxsize:
        operations = np.append(operations, [4])
    return operations