from opendomain_utils.transform_genotype import transform_Genotype
import subprocess
import numpy as np
import os
import pickle as pkl

def encode_train(adj, obj, save_path='EXP'):
    genotype = transform_Genotype(adj, obj)
    cmd = ['python', f'{os.getcwd()}/train.py', '--arch={}'.format(genotype),
           '--save={}'.format(save_path)]
    train = subprocess.Popen(cmd)
    train.wait()

def read_results(genotype):
    '''
    reading trained results
    :arg genotype of model
    :return trained results[genotype, valid_acc]
    '''
    from opendomain_utils.genotypes import Genotype
    with open(os.path.join(f"{os.getcwd()}/trained_results/models",genotype,"performance.pkl"), 'rb') as f:
        results = pkl.load(f)
        return results

if __name__ == '__main__':
    A = np.array([
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
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
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1],
    ])
    encode_train(A, ops)
