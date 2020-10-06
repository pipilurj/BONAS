import torch
def weighted_loss_acc(predicted, groundtruth):
    loss = torch.log(2*groundtruth+1)*(predicted-groundtruth)**2
    return loss.mean()
def weighted_loss_ppl(predicted, groundtruth):
    # loss = (100/groundtruth)*(predicted-groundtruth)**2
    loss = (torch.exp(200 / groundtruth) -1 ) * (predicted - groundtruth) ** 2
    return loss.mean()
def weighted_log(predicted, groundtruth):
    loss = torch.log(groundtruth+1)*(predicted-groundtruth)**2
    return loss.mean()
def weighted_linear(predicted, groundtruth):
    loss = groundtruth*(predicted-groundtruth)**2
    return loss.mean()
def weighted_exp(predicted, groundtruth):
    loss = torch.exp(groundtruth-1)*(predicted-groundtruth)**2
    return loss.mean()