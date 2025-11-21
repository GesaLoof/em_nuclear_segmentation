import torch

def binarize_logits(logits, thresh=0.5):
    # logits: [N,1,H,W] (raw)
    probs = torch.sigmoid(logits)
    return (probs >= thresh).to(logits.dtype)  # [N,1,H,W]

def pixel_accuracy(preds_bin, targets_bin):
    # both in {0,1}, shape [N,1,H,W]
    correct = (preds_bin == targets_bin).float().mean()
    return correct

def dice_coefficient(preds_bin, targets_bin, eps=1e-7):
    # per-batch Dice over all pixels
    intersection = (preds_bin * targets_bin).sum()
    union = preds_bin.sum() + targets_bin.sum()
    dice = (2 * intersection + eps) / (union + eps)
    return dice

def iou_score(preds_bin, targets_bin, eps=1e-7):
    intersection = (preds_bin * targets_bin).sum()
    union = (preds_bin + targets_bin).clamp(max=1).sum()
    iou = (intersection + eps) / (union + eps)
    return iou