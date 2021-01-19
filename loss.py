from torch.utils.data import Dataset, DataLoader
from torch.nn import BCELoss

import torch.optim as optim
import torch
    

def get_l2_loss(segmented_predicted, ground_truth):
    loss = ground_truth - segmented_predicted.norm()
    return loss


def get_segmentation_loss(segmented_predicted, ground_truth):
    bce_loss = BCELoss()(segmented_predicted, ground_truth)
    dice_loss = 2 * (segmented_predicted * ground_truth).sum() / (torch.sum(segmented_predicted)+torch.sum(ground_truth))
    dice_loss = dice_loss.clamp(1e-7)
    loss = bce_loss - torch.log(dice_loss)
    return loss