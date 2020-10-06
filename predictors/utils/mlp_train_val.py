import torch
import numpy as np
from scipy.stats import spearmanr

def train( model, optimizer, loss, train_loader, epoch, logging=None):
    total_loss_train = 0
    count = 0
    total_difference = 0
    total_corr = 0
    predicted = []
    ground_truth = []
    for i_batch, sample_batched in enumerate(train_loader):
        architecture = sample_batched['architecture']
        accuracys = sample_batched['accuracy'].view(-1, 1)
        architecture = architecture.cuda()
        accuracys = accuracys.cuda()
        model.train()
        optimizer.zero_grad()
        outputs = model(architecture)
        loss_train = loss(outputs, accuracys)
        loss_train.backward()
        optimizer.step()
        difference = outputs - accuracys
        difference = torch.abs(difference)
        difference = torch.mean(difference, 0)
        total_difference += difference.item()
        total_loss_train += loss_train.item()
        count += 1
        vx = outputs.cpu().detach().numpy().flatten()
        vy = accuracys.cpu().detach().numpy().flatten()
        predicted.append(vx)
        ground_truth.append(vy)
    predicted = np.hstack(predicted)
    ground_truth = np.hstack(ground_truth)
    corr, p = spearmanr(predicted, ground_truth)
    if (logging):
        logging.info("epoch {:d}".format(epoch + 1) + " train results:" + "train loss= {:.6f}".format(
            total_loss_train / count) + "abs_error:{:.6f}".format(total_difference / count) + "corr:{:.6f}".format(
            corr))
    else:
        print("epoch {:d}".format(epoch + 1) + " train results:" + "train loss= {:.6f}".format(
            total_loss_train / count) + "abs_error:{:.6f}".format(total_difference / count) + "corr:{:.6f}".format(
            corr))


def validate(model, loss, validation_loader, logging=None):
    loss_val = 0
    overall_difference = 0
    total_corr = 0
    count = 0
    predicted = []
    ground_truth = []
    for i_batch, sample_batched in enumerate(validation_loader):
        architecture = sample_batched['architecture']
        accuracys = sample_batched['accuracy'].view(-1, 1)
        architecture = architecture.cuda()
        accuracys = accuracys.cuda()
        model.eval()
        outputs = model(architecture)
        loss_eval = loss(outputs, accuracys)
        difference = outputs - accuracys
        difference = torch.abs(difference)
        difference = torch.mean(difference, 0)
        loss_val += loss_eval.item()
        overall_difference += difference.item()
        vx = outputs.cpu().detach().numpy().flatten()
        vy = accuracys.cpu().detach().numpy().flatten()
        predicted.append(vx)
        ground_truth.append(vy)
        count += 1
    predicted = np.hstack(predicted)
    ground_truth = np.hstack(ground_truth)
    corr, p = spearmanr(predicted, ground_truth)
    if (logging):
        logging.info(" test results:" + "loss= {:.6f}".format(
            loss_val / count) + "abs_error:{:.6f}".format(overall_difference / count) + "corr:{:.6f}".format(corr))
    else:
        print(" test results:" + "loss= {:.6f}".format(
            loss_val / count) + "abs_error:{:.6f}".format(overall_difference / count) + "corr:{:.6f}".format(corr))
    return corr, overall_difference / count, loss_val / count