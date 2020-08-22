import torch
import numpy as np
import torch.optim as optim 
from tqdm import tqdm
import torch.nn.functional as F
from mmd import mmd
from groupLasso import groupLasso
from metrics import MetricTracker, Precision_score, Recall_score, F1_score, F2_score, Hamming_loss, Subset_accuracy, \
            Accuracy_score, One_error, Coverage_error, Ranking_loss, LabelAvgPrec_score

def loss_fun(mloss, y, logits1, logits2, batch_size, l2_logits1, l2_logits12, sigma, swap_rate, lambda2, lambda3):
    
    loss_array_1 = mloss(logits1, y, reduction='none')
    loss_array_2 = mloss(logits2, y, reduction='none')

    # Group Lasso
    # Error loss array is the average of group lassos of extra and missing class labels.
    # Classes are the indexes of potential missing or extra classes for every sample.
    error_loss_array_1, classes_1 = groupLasso(y, logits_1)
    error_loss_array_2, classes_2 = groupLasso(y, logits_2)
    
    alpha = 0.2
    brutto_loss_array_1 = loss_array_1 + alpha * error_loss_array_1
    brutto_loss_array_2 = loss_array_2 + alpha * error_loss_array_2

    L2 = mmd(l2_logits_m1, l2_logits_m2, sigma) * lambda2

    L3 = mmd(logits_1, logits_2, sigma) * lambda3
    
    # Chooses the args of the (batch_size*1/4) low loss samples in the corresponding low_loss arrays
    low_loss_args_1 = torch.argsort(brutto_loss_array_1)[:int(batch_size * swap_rate)]
    low_loss_args_2 = torch.argsort(brutto_loss_array_2)[:int(batch_size * swap_rate)]
    # Gets the low_loss_samples as conducted by the peer network
    low_loss_samples_1 = torch.gather(loss_array_1, 0, low_loss_args_2)
    low_loss_samples_2 = torch.gather(loss_array_2, 0, low_loss_args_1)

    loss_1 = torch.mean(low_loss_samples_1)
    loss_2 = torch.mean(low_loss_samples_2)

    return loss_1+L3-L2, loss_2+L3-L2, L3, L2    

def run_together(net1, net2, train_data_loader, val_data_loader, epochs, batch_size, sigma, swap_rate, lambda2, lambda3):

    optimizer1 = optim.Adam(net1.parameters())
    optimizer2 = optim.Adam(net2.parameters())
    lossTracker1 = MetricTracker()
    lossTracker2 = MetricTracker()
    mloss = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        
        print(f"epoch: {epoch}")
        running_loss = 0.0

        for idx, data in enumerate(tqdm(train_data_loader, desc='training')):

            numSample = data["bands10"].size(0)
            bands = torch.cat((data["bands10"], data["bands20"]), dim=1).to(torch.device("cuda"))
            labels = data["label"].to(torch.device("cuda"))
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            logits_1, l2_logits1 = net1(bands)
            logits_2, l2_logits2 = net2(bands)

            loss1, loss2 = loss_fun(mloss, labels, logits_1, logits_2, batch_size, l2_logits1, l2_logits12, sigma, swap_rate, lambda2, lambda3)
            loss1.backward()
            loss2.backward()
            optimizer1.step()
            optimizer2.step()

            lossTracker1.update(loss1.item(), numSample)
            lossTracker2.update(loss2.item(), numSample)

            print('Train loss1: {:.6f}'.format(lossTracker1.avg))
            print('Train loss2: {:.6f}'.format(lossTracker2.avg))
        
            break
        break

