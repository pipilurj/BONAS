import torch
import numpy as np
from scipy.stats import spearmanr
import logging

def train(model, optimizer, loss, train_loader, epoch):
    logging.info("training gcn ... ")
    total_loss_train = 0
    count = 0
    total_difference = 0
    predicted = []
    ground_truth = []
    model.train()
    for i_batch, sample_batched in enumerate(train_loader):
        adjs, features, accuracys = sample_batched['adjacency_matrix'], sample_batched['operations'], \
                                    sample_batched['accuracy'].view(-1, 1)
        adjs, features, accuracys = adjs.cuda(), features.cuda(), accuracys.cuda()
        optimizer.zero_grad()
        outputs = model(features, adjs)
        loss_train = loss(outputs, accuracys)
        loss_train.backward()
        optimizer.step()
        count += 1
        difference = torch.mean(torch.abs(outputs - accuracys), 0)
        total_difference += difference.item()
        total_loss_train += loss_train.item()
        vx = outputs.cpu().detach().numpy().flatten()
        vy = accuracys.cpu().detach().numpy().flatten()
        predicted.append(vx)
        ground_truth.append(vy)
    predicted = np.hstack(predicted)
    ground_truth = np.hstack(ground_truth)
    corr, p = spearmanr(predicted, ground_truth)
    logging.info("epoch {:d}".format(epoch + 1) + " train results:" + "train loss= {:.6f}".format(
        total_loss_train / count) + "abs_error:{:.6f}".format(total_difference / count) + "corr:{:.6f}".format(
        corr))


def validate(model, loss, validation_loader, logging=None):
    loss_val = 0
    overall_difference = 0
    count = 0
    predicted = []
    ground_truth = []
    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(validation_loader):
            adjs, features, accuracys = sample_batched['adjacency_matrix'], sample_batched['operations'], \
                                        sample_batched['accuracy'].view(-1, 1)
            adjs, features, accuracys = adjs.cuda(), features.cuda(), accuracys.cuda()
            outputs = model(features, adjs)
            loss_train = loss(outputs, accuracys)
            count += 1
            difference = torch.mean(torch.abs(outputs - accuracys), 0)
            overall_difference += difference.item()
            loss_val += loss_train.item()
            vx = outputs.cpu().detach().numpy().flatten()
            vy = accuracys.cpu().detach().numpy().flatten()
            predicted.append(vx)
            ground_truth.append(vy)
        predicted = np.hstack(predicted)
        ground_truth = np.hstack(ground_truth)
        corr, p = spearmanr(predicted, ground_truth)
    logging.info("test result " + " loss= {:.6f}".format(loss_val / count) + " abs_error:{:.6f}".format(
        overall_difference / count) + " corr:{:.6f}".format(corr))
    return corr, overall_difference / count, loss_val / count
