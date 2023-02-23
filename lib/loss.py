
import torch
from torch import nn
import numpy as np

bn = lambda x:( x - torch.mean(x,dim=0,keepdim=True))/(torch.std(x,dim=0,keepdim=True)+1e-6)
bn_np = lambda x:( x - np.mean(x,axis=0,keepdims=True))/(np.std(x,axis=0,keepdims=True)+1e-6)
bn2 = lambda x,m,s:( x - m)/(s+1e-6)

def byol_loss(x,y):
    x = bn(x)
    y = bn(y)
    x = nn.functional.normalize(x,dim=-1)
    y = nn.functional.normalize(y,dim=-1)
    return ((x-y)**2).sum(-1).mean(0)

def sim_clr_loss(x,y):
    x = nn.functional.normalize(x,dim=-1)
    y = nn.functional.normalize(y,dim=-1)
    cos = torch.einsum('ac,bc->ab',x,y) / 0.1
    loss = nn.functional.cross_entropy(cos,torch.LongTensor(np.arange(len(x))).cuda())
    return loss

def our_loss(x,y,k):
    x = nn.functional.normalize(x,dim=-1)
    y = nn.functional.normalize(y,dim=-1)
    dx = torch.einsum('ac,bc->ab',x,x) / 0.1
    dy = torch.einsum('ac,bc->ab',y,y)  / 0.1
    dy = torch.exp(k)* torch.softmax(dy,dim=-1)
    loss = nn.functional.cross_entropy(dx,dy.detach().cuda())
    return loss

def match_loss_fn(x,y,k=None,loss_type='ours'):
    if loss_type ==' byol':
        return byol_loss(x,y)
    if loss_type == 'ours':
        return our_loss(x,y,k)
    if loss_type == 'ours':
        return our_loss(x,y,k)
    else:
        raise AssertionError(f"Loss {loss_type} Not Implemented")