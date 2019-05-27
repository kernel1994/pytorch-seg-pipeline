import torch
import torch.nn as nn


def dice_score(pred, target, smooth=1.):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()

    score = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return score


def dice_loss(pred, target, smooth=1.):
    loss = 1. - dice_score(pred, target)

    return loss


def dice_loss_multiclass(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1. - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()
