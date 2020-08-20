import torch
import numpy as np

def loss_fun(y, logits1, logits2, batch_size, l2_logits1, l2_logits12, sigma, swap_rate, lambda2, lambda3):
    optimizer = optim.Adam(net.parameters)
    ce = F.cross_entropy()

def run_together(net1, net2, epochs, batch_size, sigma, swap_rate, lambda2, lambda3):

    

    for epoch in range(num_epochs):

        running_loss = 0.0

