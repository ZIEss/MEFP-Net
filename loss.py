import torch.nn as nn
import torch
from pytorch_lightning.metrics import F1
import torch.nn.functional as F
from pytorch_lightning.metrics import ConfusionMatrix
import numpy as np

cfs = ConfusionMatrix(3)


class DiceLoss_multiple(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_multiple, self).__init__()
        self.sfx = nn.Softmax(dim=1)

    def binary_dice(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

    def forward(self, ipts, gt):
        ipts = self.sfx(ipts)
        c = ipts.shape[1]
        sum_loss = 0
        for i in range(c):
            tmp_inputs = ipts[:, i]
            tmp_gt = gt[:, i]
            tmp_loss = self.binary_dice(tmp_inputs, tmp_gt)
            sum_loss += tmp_loss
        return sum_loss / c


class IoU_multiple(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_multiple, self).__init__()
        self.sfx = nn.Softmax(dim=1)

    def forward(self, inputs, targets, smooth=1):
        inputs = self.sfx(inputs)
        c = inputs.shape[1]
        inputs = torch.max(inputs, 1).indices.cpu()
        targets = torch.max(targets, 1).indices.cpu()
        cfsmat = cfs(inputs, targets).numpy()

        sum_iou = 0
        for i in range(c):
            tp = cfsmat[i, i]
            fp = np.sum(cfsmat[0:3, i]) - tp
            fn = np.sum(cfsmat[i, 0:3]) - tp

            tmp_iou = tp / (fp + fn + tp)
            sum_iou += tmp_iou

        return sum_iou / c


class DiceLoss_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class IoU_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU


# 组合
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def forward(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return 0.5 * a + 0.5 * b


# 组合2
# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input, target):
#         bce = F.binary_cross_entropy_with_logits(input, target)
#         # bce = nn.BCEWithLogitsLoss(input, target)
#         smooth = 1e-4
#         input = torch.sigmoid(input)
#         num = target.size(0)
#         input = input.view(num, -1)
#         target = target.view(num, -1)
#         intersection = (input * target)
#         dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#         return 0.5 * bce + dice

# 组合3
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        # bce = nn.BCEWithLogitsLoss(input, target)
        smooth = 1e-4
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.3 * bce + 0.7 * dice
