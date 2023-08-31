#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import torch
import logging
from torch import nn
from d2l import torch as d2l


# In[ ]:


loss = nn.CrossEntropyLoss(reduction="none")


# In[2]:


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# In[ ]:


def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')


# In[ ]:


def accuracy(y_hat, y):  
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# In[ ]:


def evaluate_accuracy(net, data_iter, devices):  
    if isinstance(net, torch.nn.Module):
        net.eval()  
    metric = d2l.Accumulator(2)  
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(devices[0]), y.to(devices[0])
            metric.add(accuracy(net(X), y), y.numel())
    return (metric[0] / metric[1])


# In[ ]:


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay, file_number):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss','train acc', 'valid acc']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    log_path = './log/exp' + file_number + '.log'
    logger = get_logger(log_path)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            train_acc = accuracy(output, labels)/labels.numel()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], train_acc, None, None))
        train_loss = metric[0] / metric[1]
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_acc = evaluate_accuracy(net, valid_iter, devices)
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, None, valid_acc, valid_loss.detach().cpu()))
            logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t train_acc={:.3f} vali_loss={:.5f}\t vali_acc={:.3f}'.format(epoch + 1 , num_epochs, train_loss, train_acc, valid_loss, valid_acc ))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f', train accuracy :{train_acc}' + f', valid accuracy :{valid_acc}' + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
    logger.handlers.clear()

