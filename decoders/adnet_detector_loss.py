import sys

import torch
import torch.nn as nn


class SegDetectorLossBuilder():
    '''
    Build loss functions for SegDetector.
    Details about the built functions:
        Input:
            pred: A dict which contains predictions.
                thresh: The threshold prediction
                binary: The text segmentation prediction.
                thresh_binary: Value produced by `step_function(binary - thresh)`.
            batch:
                gt: Text regions bitmap gt.
                mask: Ignore mask,
                    pexels where value is 1 indicates no contribution to loss.
                thresh_mask: Mask indicates regions cared by thresh supervision.
                thresh_map: Threshold gt.
        Return:
            (loss, metrics).
            loss: A scalar loss value.
            metrics: A dict contraining partial loss values.
    '''

    def __init__(self, loss_class, *args, **kwargs):
        self.loss_class = loss_class
        self.loss_args = args
        self.loss_kwargs = kwargs

    def build(self):
        return getattr(sys.modules[__name__], self.loss_class)(*self.loss_args, **self.loss_kwargs)


class DilateLoss(nn.Module):
    def __init__(self, eps=1e-6, l1_scale=5, bce_scale=1, dice_scale=5):
        super(DilateLoss, self).__init__()
        self.l1_loss = DilateL1Loss()
        self.l1_scale = l1_scale
        self.bce_loss = BalanceCrossEntropyLoss()
        self.bce_scale = bce_scale
        self.dice_loss = DiceLoss(eps=eps)
        self.dice_scale = dice_scale

    def forward(self, pred, batch):
        metrics = {}
        bce_loss = self.bce_loss(pred['origin'], batch['gt_origin'], batch['mask'])
        metrics['bce_loss'] = bce_loss
        dice_loss = self.dice_loss(pred['shrink'], batch['gt_shrink'], batch['mask'])
        metrics['dice_loss'] = dice_loss
        l1_loss = self.l1_loss(pred['dilate'], batch['dilate'])
        metrics['l1_loss'] = l1_loss

        loss = self.bce_scale * bce_loss + self.dice_scale * dice_loss + self.l1_scale * l1_loss
        return loss, metrics

class DilateL1Loss(nn.Module):
    def __init__(self):
        super(DilateL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, gt):
        eps = 1e-6
        mask = gt > 0
        mask = mask.float()
        loss = (torch.abs(pred[:, 0] - gt) * mask).sum() / (mask.sum() + eps)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        assert pred.dim() == 4, pred.dim()
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss

class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                return_origin=False):
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()),
                            int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(
            pred, gt, reduction='none')[:, 0, :, :]
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) /\
            (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss