import torch

def dice_coefficient(preds, targets, num_classes, epsilon=1e-6):
    """preds, targets: [N, H, W]; returns list of class Dice sums over batch"""
    dice_per_class = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum(dim=(1,2))
        union = pred_c.sum(dim=(1,2)) + target_c.sum(dim=(1,2))
        dice = (2 * intersection + epsilon) / (union + epsilon)
        dice[union == 0] = 1.0
        dice_per_class.append(dice.sum().item())
    return dice_per_class

def dice_coefficient_batch(preds, targets, num_classes, epsilon=1e-6):
    """Returns [C] tensor of mean Dice per class over batch"""
    dice_scores = []
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).sum(dim=(1,2)).float()
        pred_sum = pred_c.sum(dim=(1,2)).float()
        target_sum = target_c.sum(dim=(1,2)).float()
        union = pred_sum + target_sum
        dice = (2 * intersection + epsilon) / (union + epsilon)
        dice[union == 0] = 1.0
        dice_scores.append(dice)
    dice_scores = torch.stack(dice_scores, dim=1)  # [N, C]
    return dice_scores.mean(dim=0)

def iou_score(preds, targets, num_classes, epsilon=1e-6):
    iou_per_class = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        inter = (pred_c * target_c).sum(dim=(1,2))
        union = pred_c.sum(dim=(1,2)) + target_c.sum(dim=(1,2)) - inter
        iou = (inter + epsilon) / (union + epsilon)
        iou[union == 0] = 1.0
        iou_per_class.append(iou.sum().item())
    return iou_per_class

def precision_score(preds, targets, num_classes, epsilon=1e-6):
    prec = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        tp = (pred_c * target_c).sum(dim=(1,2))
        fp = (pred_c * (1 - target_c)).sum(dim=(1,2))
        prec_val = (tp + epsilon) / (tp + fp + epsilon)
        prec.append(prec_val.sum().item())
    return prec

def recall_score(preds, targets, num_classes, epsilon=1e-6):
    rec = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        tp = (pred_c * target_c).sum(dim=(1,2))
        fn = ((1 - pred_c) * target_c).sum(dim=(1,2))
        rec_val = (tp + epsilon) / (tp + fn + epsilon)
        rec.append(rec_val.sum().item())
    return rec

def f1_score(preds, targets, num_classes, epsilon=1e-6):
    f1 = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        tp = (pred_c * target_c).sum(dim=(1,2))
        fp = (pred_c * (1 - target_c)).sum(dim=(1,2))
        fn = ((1 - pred_c) * target_c).sum(dim=(1,2))
        prec = (tp + epsilon) / (tp + fp + epsilon)
        rec = (tp + epsilon) / (tp + fn + epsilon)
        f1_val = 2 * prec * rec / (prec + rec + epsilon)
        f1.append(f1_val.sum().item())
    return f1