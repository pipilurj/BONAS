import numpy as np
from opendomain_utils.genotypes import Genotype, PRIMITIVES
from settings import OPS

def transform_Genotype(adj, ops):
    adj = np.delete(adj, 2, axis=0)
    adj = np.delete(adj, 3, axis=0)
    adj = np.delete(adj, 4, axis=0)
    adj = np.delete(adj, 5, axis=0)
    cell = [
        (OPS[np.nonzero(ops[2])[0][0]], np.nonzero(adj[:, 2])[0][0]),
        (OPS[np.nonzero(ops[3])[0][0]], np.nonzero(adj[:, 3])[0][0]),
        (OPS[np.nonzero(ops[4])[0][0]], np.nonzero(adj[:, 4])[0][0]),
        (OPS[np.nonzero(ops[5])[0][0]], np.nonzero(adj[:, 5])[0][0]),
        (OPS[np.nonzero(ops[6])[0][0]], np.nonzero(adj[:, 6])[0][0]),
        (OPS[np.nonzero(ops[7])[0][0]], np.nonzero(adj[:, 7])[0][0]),
        (OPS[np.nonzero(ops[8])[0][0]], np.nonzero(adj[:, 8])[0][0]),
        (OPS[np.nonzero(ops[9])[0][0]], np.nonzero(adj[:, 9])[0][0]),
    ]
    ft_model = Genotype(
        normal=cell,
        normal_concat=[2, 3, 4, 5],
        reduce=cell,
        reduce_concat=[2, 3, 4, 5]
    )
    return ft_model

def transform_matrix(genotype):
    normal = genotype.normal
    node_num = len(normal)+3
    adj = np.zeros((node_num, node_num))
    ops = np.zeros((node_num, len(OPS)))
    for i in range(len(normal)):
        op, connect = normal[i]
        if connect == 0 or connect==1:
            adj[connect][i+2] = 1
        else:
            adj[(connect-2)*2+2][i+2] = 1
            adj[(connect-2)*2+3][i+2] = 1
        ops[i+2][OPS.index(op)] = 1
    adj[2:-1, -1] = 1
    ops[0:2, 0] = 1
    ops[-1][-1] = 1
    return adj, ops

def geno_to_archs(genotypes, ei_scores=None):
    # print(genotypes)
    archs = []
    for i in range(len(genotypes)):
        if isinstance(genotypes, str):
            adj, op = transform_matrix(eval(genotypes[i]))
        else:
            adj, op = transform_matrix(genotypes[i])
        if ei_scores:
            datapoint = {'adjacency_matrix': adj, 'operations': op, 'metrics': ei_scores[i]}
        else:
            datapoint = {'adjacency_matrix': adj, 'operations': op}
        archs.append(datapoint)
    return archs

def geno2mask(genotype):
    des = -1
    mask = np.zeros(14, 8)
    op_names, indices = zip(*genotype.normal)
    for cnt, (name, index) in enumerate(zip(op_names, indices)):
        if cnt % 2 == 0:
            des += 1
            total_state = sum(i+2 for i in range(des))
        op_idx = PRIMITIVES.index(name)
        node_idx = index + total_state
        mask[node_idx, op_idx] = 1
    print(mask)
    return mask

if __name__ == '__main__':
    A = np.array([
        [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    ops = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
    ])
    geno = transform_Genotype(A, ops)
    print(geno.normal)
    # print(geno.ops)
    # print(transform_matrix(geno) - A)
    print(transform_matrix(geno)[1] - ops)
