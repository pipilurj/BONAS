import random
import numpy as np
import time
from settings import OPS

def generate_adj():
    mat = np.zeros([11, 11])
    mat[:, 10] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    a = random.choice([0, 1])
    b = random.choice([0, 1])
    c = random.choice([0, 1, [2,3]])
    d = random.choice([0, 1, [2,3]])
    e = random.choice([0, 1, [2,3], [4,5]])
    f = random.choice([0, 1, [2,3], [4,5]])
    g = random.choice([0, 1, [2,3], [4,5],[6,7]])
    h = random.choice([0, 1, [2,3], [4,5],[6,7]])
    mat[a, 2] = 1
    mat[b, 3] = 1
    mat[c, 4] = 1
    mat[d, 5] = 1
    mat[e, 6] = 1
    mat[f, 7] = 1
    mat[g, 8] = 1
    mat[h, 9] = 1
    return mat


def generate_ops():
    op_num = len(OPS)
    op_matrix = np.zeros((11, op_num))
#     op_matrix = np.zeros((11, 6))
    op_matrix[0][0] = 1
    op_matrix[1][0] = 1
    for i in range(8):
        idx = random.choice(list(range(1, op_num-1)))
        op_matrix[i + 2][idx] = 1
    op_matrix[10][-1] = 1
    return op_matrix

def generate_archs(generate_num):
    archs = []
    archs_hash = []
    cnt = 0
    while cnt < generate_num:
        adj = generate_adj()
        ops = generate_ops()
        if is_valid(adj, ops):
            arch = {"adjacency_matrix":adj, "operations":ops}
            arch_hash = str(hash(str(arch)))
            if arch_hash not in archs_hash:
                archs.append(arch)
                archs_hash.append(arch_hash)
                cnt += 1
    return archs

def is_valid(adj, op, step=4):
    for i in range(step):
        if (adj[:, 2 * i + 2] == adj[:, 2 * i + 3]).all() and (op[2 * i + 2, :] == op[2 * i + 3, :]).all():
            return 0
    return 1

if __name__ =='__main__':
    t1 = time.time()
    arch = generate_archs(10000)