import torch
import torch.nn as nn

def cross_entropy(input, target):
    return - torch.mean(torch.sum(target * torch.log(input) + (1 - target) * torch.log(1 - input), 1))

def L1_loss(input,target):
    return torch.mean(torch.abs(target - input))

class Seg_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,gt_map,pred_map):
        return cross_entropy(gt_map,pred_map)

class Rectification_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,gt_map,pred_map):
        return L1_loss(gt_map, pred_map)
