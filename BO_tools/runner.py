import time
from BO_tools import optimizer_gcn
import random
import argparse
import logging
from opendomain_utils.encode_and_train import encode_train, read_results
from opendomain_utils.transform_genotype import transform_Genotype, transform_matrix, geno2mask
from data_generators.dynamic_generate import generate_archs
from opendomain_utils.ioutils import get_geno_hash, get_trained_csv, get_trained_archs, update_trained_csv, \
    update_trained_pickle
from samplers import EASampler, RandomSampler
import numpy as np
from BO_tools.trainer import Trainer


class Runner(object):
    def __init__(self,
                 gcn_epochs=100,
                 gcn_lr=0.001,
                 loss_num=3,
                 generate_num=1000,
                 iterations=1000,
                 sample_method="random",
                 training_cfg=None,
                 bo_sample_num=5,
                 mode="supernet",
                 if_init_samples=True,
                 init_num=10,
                 eval_submodel_path = None
                 ):
        assert training_cfg is not None
        self.mode = mode
        self.generate_num = generate_num
        self.max_acc = 0
        self.max_hash = 'none'
        self.gcn_lr = gcn_lr
        self.if_init_samples = if_init_samples
        self.bo_sample_num = bo_sample_num
        self.eval_submodel_path = eval_submodel_path
        self.loss_num = loss_num
        self.gcn_epochs = gcn_epochs
        self.iterations = iterations
        self.sample_method = sample_method
        self.trainer = Trainer(**training_cfg)
        self.train_init_samples(if_init_samples, init_num)
        self.build_optimizer()
        self.build_sampler()

    def train_init_samples(self, if_init_samples, init_num):
        if not if_init_samples:
            return
        original_graphs = generate_archs(generate_num=self.generate_num)
        self.trained_arch_list = get_trained_archs()
        logging.info("********************** training init samples **********************")
        while len(self.trained_arch_list) < init_num:
            init_samples = random.sample(original_graphs, self.bo_sample_num)
            self.train_super_model(init_samples)
            self.trained_arch_list = get_trained_archs()
        logging.info("********************** finished init training **********************")

    def build_optimizer(self):
        # builder BO optimizer
        self.trained_arch_list = get_trained_archs()
        self.optimizer = optimizer_gcn.Optimizer(self.trained_arch_list, train_epoch=self.gcn_epochs,
                                                 lr=self.gcn_lr, lossnum=self.loss_num)
        self.optimizer.train()

    def build_sampler(self):
        # build sampler
        # TODO: later add RL controller if needed
        if self.sample_method == 'random':
            print(self.generate_num)
            self.sampler = RandomSampler(generate_num=self.generate_num)
        elif self.sample_method == 'ea':
            self.sampler = EASampler(self.trained_arch_list)
        else:
            raise NotImplementedError(f"{self.sample_method} is not a valid sampler type,"
                                      f"currently only support EA and Random")

    def train_one_model(self, data_point):
        '''
        :param data_point: an architecture[adj,ops]
        :return:trained acc result
        trains an architecture and saves result to csv and pickle file
        '''
        self.trained_arch_list = get_trained_archs()
        trained_models = get_geno_hash(self.trained_arch_list)
        adj, ops = data_point['adjacency_matrix'], data_point['operations']
        genohash = str(hash(str(transform_Genotype(adj, ops))))
        if genohash not in trained_models:
            encode_train(adj=adj, obj=ops, save_path=genohash)
            results = read_results(genohash)
            data_point = {'adjacency_matrix': adj, "operations": ops,
                          "metrics": results[1],
                          "genotype": str(results[0]), "hash": genohash}
            self.trained_arch_list = update_trained_pickle(
                data_point)
            update_trained_csv(
                dict(genotype=str(results[0]), hashstr=genohash, acc=results[1]))
            if hasattr(self, 'optimizer'):
                self.optimizer.update_data(self.trained_arch_list)
            if hasattr(self, 'sampler'):
                self.sampler.update_sampler(data_point, ifappend=True)
            if results[1] > self.max_acc:
                self.max_acc = results[1]
                self.max_hash = genohash

    def train_super_model(self, data_points):
        '''
                :param data_points: list<dict<adj,ops>>
                :return:trained accs and hashes of subnets
                '''
        genotypes = [transform_Genotype(data_point['adjacency_matrix'], data_point['operations']) for data_point in
                     data_points]
        eval_genotypes = None
        genohashs = [hash(str(genotype)) for genotype in genotypes]
        eval_num = len(genohashs)
        if self.eval_submodel_path:
            import pickle
            with open(self.eval_submodel_path, 'rb') as f:
                datapoints = pickle.load(f)
                eval_genotypes = [datapoint['genotype'] for datapoint in datapoints]
                eval_num = len(eval_genotypes)
        results = self.trainer.train_and_eval(genotypes, eval_genotypes)
        accs = [result[1] for result in results]
        trained_archs = []
        trained_csvs = []
        trained_datapoints = []
        for i in range(eval_num):
            trained_arch = {'adjacency_matrix': data_points[i]['adjacency_matrix'],
                            "operations": data_points[i]['operations'],
                            "metrics": accs[i],
                            "genotype": str(genotypes[i]), "hash": genohashs[i]}
            trained_csv = {
                "genotype": str(genotypes[i]), 'hashstr': genohashs[i], 'acc': accs[i]
            }
            trained_archs.append(trained_arch)
            trained_csvs.append(trained_csv)
            if accs[i] > self.max_acc:
                self.max_acc = accs[i]
                self.max_hash = genohashs[i]
            trained_datapoints.append(trained_arch)
        self.trained_arch_list = update_trained_pickle(
            trained_archs)
        update_trained_csv(trained_csvs)
        if hasattr(self, 'optimizer'):
            self.optimizer.update_data(self.trained_arch_list)
        if hasattr(self, 'sampler'):
            self.sampler.update_sampler(trained_datapoints, ifappend=True)

    def sample(self):
        new_domain = self.sampler.sample()
        return new_domain

    def run_single(self):
        selection_index = 0
        selection_size = 0
        for iteration in range(self.iterations):
            if selection_index == selection_size:
                new_domain = self.sample()
                logging.info("Update GCN!!")
                self.optimizer.retrain_NN()
                logging.info("Update LR!!")
                self.optimizer.retrain_LR()
                selected_points, pred_acc, selected_indices = self.optimizer.select_multiple(new_domain=new_domain,
                                                                                             cap=self.bo_sample_num)
                selection_size = len(selected_points)
                selection_index = 0
                # Retrain the neural network
            logging.info(f"selection {str(selection_index)}, pred_Acc:{pred_acc[selection_index]}")
            # fully train models!!!!
            self.train_one_model(selected_points[selection_index])
            logging.info(f"iter{iteration},current max: {self.max_acc}, max hash {self.max_hash}")
            selection_index += 1

    def run_super(self):
        for iteration in range(self.iterations):
            # fully train super_model!!!!
            self.trained_arch_list = get_trained_archs()
            trained_models = get_geno_hash(self.trained_arch_list)
            new_domain = self.sample()
            selected_points, pred_acc, selected_indices = self.optimizer.select_multiple_unique(new_domain=new_domain,
                                                                                                cap=self.bo_sample_num,
                                                                                                trained_models=trained_models)
            self.train_super_model(selected_points)
            logging.info(f"iter{iteration},current max: {self.max_acc}, max hash {self.max_hash}")
            logging.info("Update GCN!!")
            self.optimizer.retrain_NN()
            logging.info("Update LR!!")
            self.optimizer.retrain_LR()

    def run(self):
        if self.mode == "supernet":
            self.run_super()
        elif self.mode == "singlenet":
            self.run_single()
        else:
            raise NotImplementedError
