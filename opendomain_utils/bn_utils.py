import copy
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_running_statistics(model, data_loader, mask):
    # Resetting statistics of subnets' BN, forwarding part of training data through each subnet to be evaluated
    bn_mean = {}
    bn_var = {}
    forward_model = copy.deepcopy(model)
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):

            bn_mean[name] = AverageMeter()
            bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    if bn.affine:
                        weight = bn.weight[:_feature_dim].to(x.device)
                        bias = bn.bias[:_feature_dim].to(x.device)
                        return F.batch_norm(
                            x, batch_mean, batch_var, weight,
                            bias, False, 0.0, bn.eps,
                        )
                    else:
                        return F.batch_norm(
                            x, batch_mean, batch_var, None,
                            None, False, 0.0, bn.eps,
                        )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    batchsize = -1
    inputsize = 1024
    input_cat = []
    for i, (input, target) in enumerate(data_loader):
        if batchsize == -1:
            batchsize = input.size(0)
        if len(input_cat) * batchsize < inputsize:
            input_cat.append(input)
            continue
        else:
            input = torch.cat(input_cat)
            input_cat = []
        with torch.no_grad():
            forward_model(input.cuda(), mask)
        if (i + 1) * batchsize > 10000:
            break

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)
