import numpy as np

def add_global_node( mx, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if (ifAdj):
        mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        np.fill_diagonal(mx, 1)
        mx = mx.T
    else:
        mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
    return mx

def padzero( mx, ifAdj, maxsize=7):
    if ifAdj:
        while mx.shape[0] < maxsize:
            mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
    else:
        while mx.shape[0] < maxsize:
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
    return mx

def net_decoder( operations):
    operations = np.array(operations, dtype=np.int32)
    for i in range(len(operations)):
        if operations[i] == 2: #input
            operations[i]=3
        elif operations[i] == 3: #conv1
            operations[i]=0
        elif operations[i] == 4: #pool
            operations[i]=2
        elif operations[i] == 5: #conv3
            operations[i]=1
        elif operations[i] == 6: #output
            operations[i]=4
    one_hot = np.zeros((len(operations), 5))
    one_hot[np.arange(len(operations)), operations] = 1
    return one_hot

if __name__=="__main__":
    # a = np.array([[0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
    #    [0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0.],
    #    [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1.],
    #    [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1.],
    #    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    print("buya")
    a = np.array([[0., 0., 1., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],])
    print(a.shape)
    a = padzero(np.array(a), True, maxsize=11)
    print(a.shape)
    # add_global_node(padzero(np.array(a['adjacency_matrix']), True, maxsize=11), True)
