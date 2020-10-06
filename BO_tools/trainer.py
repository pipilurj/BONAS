import numpy as np
import torch
import opendomain_utils.training_utils as utils
import logging
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import random
from super_model import Network, geno2mask, merge
from opendomain_utils.ioutils import copy_log_dir
from opendomain_utils.genotypes import Genotype
from settings import distributed
DataParallel = torch.nn.parallel.DistributedDataParallel if distributed else torch.nn.DataParallel
from itertools import cycle, islice
from opendomain_utils.bn_utils import set_running_statistics
import copy
CIFAR_CLASSES = 10

class Trainer:
    def __init__(self,
                 train_supernet_epochs=5,
                 data_path='data',
                 super_batch_size=64,
                 sub_batch_size=128,
                 learning_rate=0.1,
                 momentum=0.9,
                 weight_decay=3e-4,
                 report_freq=50,
                 epochs=50,
                 init_channels=36,
                 layers=20,
                 drop_path_prob=0.2,
                 seed=0,
                 grad_clip=5,
                 parallel = False,
                 mode='uniform'
                 ):
        self.parallel = parallel
        self.train_supernet_epochs = train_supernet_epochs
        self.data_path = data_path
        self.super_batch_size = super_batch_size
        self.sub_batch_size = sub_batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.mode = mode
        self.weight_decay = weight_decay
        self.report_freq = report_freq
        self.epochs = epochs
        self.init_channels = init_channels
        self.layers = layers
        self.drop_path_prob = drop_path_prob
        self.seed = seed
        self.grad_clip = grad_clip
        self.build_dataloader()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.cuda()
        self.train_loader_super, self.train_loader_sub, self.valid_loader = self.build_dataloader()

    def build_dataloader(self):
        train_transform, valid_transform = utils._data_transforms_cifar10()
        train_data = dset.CIFAR10(root=self.data_path, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=self.data_path, train=False, download=True, transform=valid_transform)
        train_loader_super = torch.utils.data.DataLoader(
            train_data, batch_size=self.super_batch_size, shuffle=True, pin_memory=True, num_workers=32)
        train_loader_sub = torch.utils.data.DataLoader(
            train_data, batch_size=self.sub_batch_size, shuffle=True, pin_memory=True, num_workers=32)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=1024, shuffle=False, pin_memory=True, num_workers=32)
        return train_loader_super, train_loader_sub, valid_loader

    def set_seed(self):
        np.random.seed(self.seed)
        cudnn.benchmark = True
        torch.manual_seed(self.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(self.seed)

    def build_model(self, mask):
        model = Network(self.init_channels, CIFAR_CLASSES, self.layers, mask=mask)
        if self.parallel:
            if distributed:
                model = DataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
            else:
                model = DataParallel(model.cuda())
        else:
            model = model.cuda()
        return model

    def train_and_eval(self, archs, eval_archs=None):
        """
        :param archs: archs sample by parallel BO
        :return: results list<genotype, top1acc>
        """
        self.genotypes = [eval(arch) if isinstance(arch, str) else arch for arch in archs]
        self.eval_genos = None
        if eval_archs != None:
            self.eval_genos = [eval(arch) if isinstance(arch, str) else arch for arch in eval_archs]
            self.genotypes = self.genotypes + self.eval_genos
        self.subnet_masks = [geno2mask(genotype) for genotype in self.genotypes]
        self.supernet_mask = merge(self.subnet_masks)
        self.iterative_indices = list(islice(cycle(list(range(len(self.subnet_masks)))), len(self.train_loader_sub)))
        supernet = self.build_model(self.supernet_mask)
        logging.info("Training Super Model ...")
        logging.info("param size = %fMB", utils.count_parameters_in_MB(supernet))
        optimizer = torch.optim.SGD(
            supernet.parameters(),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.epochs))

        for epoch in range(self.epochs):
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            supernet.drop_path_prob = self.drop_path_prob * epoch / self.epochs
            if epoch in range(self.train_supernet_epochs):
                self.train_supernet(supernet, optimizer, epoch)
            else:
                self.train(supernet, optimizer, supernet=False)
            scheduler.step()
        logging.info("Evaluating subnets ...")
        results = self.evaluate_subnets(supernet, self.subnet_masks, self.genotypes, self.eval_genos)
        return results

    def train_supernet(self, model, optimizer, epoch):
        self.train(model, optimizer, supernet=True)
        if epoch == self.train_supernet_epochs - 1:
            val_obj, val_top1, val_top5 = self.evaluate(model, self.supernet_mask)
            logging.info('Supernet valid %e %f %f', val_obj, val_top1, val_top5)
            set_running_statistics(model, self.train_loader_super, self.supernet_mask)
            val_obj, val_top1, val_top5 = self.evaluate(model, self.supernet_mask)
            logging.info('After resetbn Supernet valid %e %f %f', val_obj, val_top1, val_top5)

        copy_log_dir()

    def evaluate_subnets(self, supernet, subnet_masks, genotypes, eval_genos=None):
        results = []
        if eval_genos:
            genotypes = eval_genos
            subnet_masks = [geno2mask(geno) for geno in genotypes]
        with torch.no_grad():
            supernet.eval()
            supernet_copy = copy.deepcopy(supernet)
            i=1
            for mask, genotype in zip(subnet_masks, genotypes):
                set_running_statistics(supernet_copy, self.train_loader_sub, mask)
                obj, top1, top5 = self.evaluate(supernet_copy, mask)
                logging.info('%s th Arch %s valid %e %f %f',str(i), str(genotype.normal), obj, top1, top5)
                results.append((genotype, top1))
                copy_log_dir()
                i+=1
        return results

    def evaluate(self, model, mask):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()
        with torch.no_grad():
            for step, (input, target) in enumerate(self.valid_loader):
                input = Variable(input).cuda()
                target = Variable(target).cuda(async=True)
                logits = model(input, mask=mask)
                loss = self.criterion(logits, target)
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data.item(), n)
                top1.update(prec1.data.item(), n)
                top5.update(prec5.data.item(), n)
        return objs.avg, top1.avg / 100, top5.avg / 100

    def train(self, model, optimizer, supernet=False):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.train()
        iterative_indices = np.random.permutation(self.iterative_indices)
        train_loader = self.train_loader_super if supernet else self.train_loader_sub
        for step, (input, target) in enumerate(train_loader):
            if self.mode == 'uniform':
                mask = self.subnet_masks[iterative_indices[step]] if not supernet else self.supernet_mask
            else:
                mask = random.choice(self.subnet_masks)
            input = Variable(input).cuda()
            target = Variable(target).cuda(async=True)
            optimizer.zero_grad()
            logits = model(input, mask=mask)
            loss = self.criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)
            if step % self.report_freq == 0:
                logging.info('supernet train %03d %e %f %f', step, objs.avg, top1.avg / 100, top5.avg / 100)
                copy_log_dir()