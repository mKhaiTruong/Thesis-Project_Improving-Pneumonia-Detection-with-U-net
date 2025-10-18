import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

def compute_iou(preds, masks, threshold=0.5, eps=1e-6):
    
    preds_bin = (preds > threshold).float()
    
    intersection = (preds_bin * masks).sum(dim=(1,2,3))
    union        = ((preds_bin + masks) > 0).float().sum(dim=(1,2,3))
    iou          = (intersection + eps) / (union + eps)
    return iou.mean().item()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.7, pos_weight=4.0, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        preds_sigmoid = torch.sigmoid(preds)
        
        intersection = (preds_sigmoid * targets).sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / \
                (preds_sigmoid.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss

class BCETverskyLoss(nn.Module):
    def __init__(self, bce_weight=0.3, tversky_weight=0.7, pos_weight=4.0, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.tversky = smp.losses.TverskyLoss(mode='binary', from_logits=True, alpha=alpha, beta=beta, smooth=smooth)
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        tversky = self.tversky(preds, targets)
        return self.bce_weight * bce + self.tversky_weight * tversky

class FocalLoss(nn.Module):
    def __init__(self, weight = None, gamma = 2, reduction = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim = 1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1-prob)**self.gamma)*log_prob,
            targets,
            weight = self.weight,
            reduction = self.reduction
        )


class FocalLoss(nn.Module):
    
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-12
        
    def forward(self, pred, mask):   
        '''
        Args:
            pred: [B, 1, H, W]
            mask: [B, 1, H, W]
        '''
        assert pred.shape == mask.shape, 'pred and mask should have the same shape'
        
        p = torch.sigmoid(pred)
        n_pos = torch.sum(mask)
        n_neg = mask.numel() - n_pos
        
        w_pos = (1-p) ** self.gamma
        w_neg = p ** self.gamma
        
        loss_pos = -self.alpha * mask * w_pos * torch.log(p + self.eps)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + self.eps)
        
        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (n_pos + n_neg + self.eps)
        return loss
        
        
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, mask):
        '''
        Args:
            pred: [B, 1, H, W]
            mask: [B, 1, H, W]
        '''
        assert pred.shape == mask.shape, 'pred and mask should have the same shape'
        
        p = torch.sigmoid(pred)
        
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice_loss
        
class MaskIoULoss(nn.Module):
    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred, mask, pred_iou):
        '''
        Args:
            pred: [B, 1, H, W]
            mask: [B, 1, H, W]
        '''
        assert pred.shape == mask.shape, 'pred and mask should have the same shape'
        
        p = torch.sigmoid(pred)
        
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask) - intersection
        
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss
        
class FocalDiceloss_IoULoss(nn.Module):
    
    def __init__(self, weight=2.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight     = weight
        self.iou_scale  = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss  = DiceLoss()
        self.mask_iou_loss = MaskIoULoss()
    
    def forward(self, pred, mask, pred_iou):
        '''
        Args:
            pred: [B, 1, H, W]
            mask: [B, 1, H, W]
        '''
        assert pred.shape == mask.shape, 'pred and mask should have the same shape'
        
        focal_loss = self.focal_loss(pred, mask)
        dice_loss  = self.dice_loss(pred, mask)
        
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.mask_iou_loss(pred, mask, pred_iou)
        loss  = loss1 + loss2 * self.iou_scale
        return loss
    
    
import cv2, torch, numpy as np

def _threshold(x, threshold: float = None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def _list_tensor(x, y):
    sigmoid = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:     x = sigmoid(x)
            
    else:
        x, y = x, y
        if x.min() < 0:     x = sigmoid(x)  
    
    return x, y

def iou(pr, gt, eps=1e-7, threshold: float = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    
    pr_ = _threshold(pr_, threshold)
    gt_ = _threshold(gt_, threshold)
    
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()

def dice(pr, gt, eps=1e-7, threshold: float = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    
    pr_ = _threshold(pr_, threshold)
    gt_ = _threshold(gt_, threshold)
    
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3])
    return ((2. * intersection + eps) / (union + eps)).cpu().numpy()
    
def SegMetrics(pred, label, metrics):
    metric_list = []
    
    if isinstance(metrics, str):    metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):     
            continue
        
        elif metric == 'iou': 
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice': 
            metric_list.append(np.mean(dice(pred, label)))
        else:
            raise ValueError(f'This -{metric}- is not recognized.')
    
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError(f'metric mistakes in calculations')
    return metric

class AverageMeter:
    
    def __init__(self):
        self.currentTotal = 0.0
        self.iterations   = 0.0
        
    def send(self, value):
        self.currentTotal += value
        self.iterations   += 1
    
    @property
    def value(self):
        if self.iterations == 0:    return 0
        else:
            return 1.0 * self.currentTotal / self.iterations
    
    def reset(self):
        self.currentTotal = 0.0
        self.iterations   = 0.0


import numpy as np

# ----------------------------- IoU & matching -----------------------------

def calculate_iou(gt, pr, form: str = "pascal_voc") -> float:
    """
    Intersection-over-Union between one GT box and one predicted box.

    Args:
        gt: 1D array-like of length 4 (GT box).
        pr: 1D array-like of length 4 (Predicted box).
        form: 'coco' -> [x, y, w, h]; 'pascal_voc' -> [x1, y1, x2, y2].

    Returns:
        IoU in [0, 1].
    """
    gt = np.asarray(gt, dtype=float).copy()
    pr = np.asarray(pr, dtype=float).copy()

    # Convert COCO to [x1, y1, x2, y2] (inclusive pixel coordinates).
    if form == "coco":
        gt[2], gt[3] = gt[0] + gt[2], gt[1] + gt[3]
        pr[2], pr[3] = pr[0] + pr[2], pr[1] + pr[3]
    elif form != "pascal_voc":
        raise ValueError(f"Unknown box format: {form}")

    # Intersection (inclusive; +1 matches the original code's convention)
    ix1 = max(gt[0], pr[0])
    iy1 = max(gt[1], pr[1])
    ix2 = min(gt[2], pr[2])
    iy2 = min(gt[3], pr[3])

    iw = ix2 - ix1 + 1
    ih = iy2 - iy1 + 1
    if iw <= 0 or ih <= 0:
        return 0.0

    inter = iw * ih

    # Areas (inclusive)
    gt_area = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    pr_area = (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1)
    union = gt_area + pr_area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def findBestMatch(gts, pred, predIdx, threshold=0.5, form='pascal_voc', ious=None) -> int:
    """
    Greedy best-match GT index for a given prediction at an IoU threshold.
    Uses/updates an IoU cache if provided to avoid recomputation.
    """
    best_iou = -np.inf
    best_idx = -1

    for gtIdx in range(len(gts)):
        if gts[gtIdx][0] < 0:  # GT already matched (marked)
            continue

        iou = -1 if ious is None else ious[gtIdx][predIdx]
        if iou < 0:
            iou = calculate_iou(gts[gtIdx], pred, form)
            if ious is not None:
                ious[gtIdx][predIdx] = iou

        if iou >= threshold and iou > best_iou:
            best_iou = iou
            best_idx = gtIdx

    return best_idx


# ----------------------------- Per-image metrics -----------------------------

def calculatePrecision(gts, preds, threshold=0.5, form='coco', ious=None) -> float:
    """
    Kaggle-style detection precision at a single IoU threshold:
        TP / (TP + FP + FN)
    where FN counts unmatched GT boxes.
    """
    tp, fp = 0, 0
    for predIdx in range(len(preds)):
        best = findBestMatch(gts, preds[predIdx], predIdx, threshold, form, ious)
        if best >= 0:
            tp += 1
            gts[best] = -1  # mark GT as matched
        else:
            fp += 1
    fn = (gts.sum(axis=1) > 0).sum()
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0


def calculateImagePrecision(gts, preds, thresholds=(0.5,), form='coco') -> float:
    """
    Mean of Kaggle-style precision across multiple IoU thresholds for one image.
    """
    if len(gts) and len(preds):
        ious = np.ones((len(gts), len(preds)), dtype=float) * -1  # cache
    else:
        ious = None

    acc = 0.0
    for th in thresholds:
        acc += calculatePrecision(gts.copy(), preds, threshold=th, form=form, ious=ious)
    return acc / max(1, len(thresholds))


def image_counts_from_matches(gts, preds, iou_threshold=0.5, form='coco', ious=None):
    """
    Count TP, FP, FN without score filtering (greedy one-to-one matching).
    """
    gts = gts.copy()
    tp, fp = 0, 0
    for predIdx in range(len(preds)):
        best = findBestMatch(gts, preds[predIdx], predIdx, iou_threshold, form, ious)
        if best >= 0:
            tp += 1
            gts[best] = -1
        else:
            fp += 1
    fn = (gts.sum(axis=1) > 0).sum()
    return tp, fp, fn


def image_counts_with_score(gts, preds, scores, score_threshold=None, iou_threshold=0.5, form='coco'):
    """
    Count TP/FP/FN after optional score filtering; also return per-pred TP labels and kept scores.
    """
    if score_threshold is not None and scores is not None:
        keep = scores >= score_threshold
        preds = preds[keep]
        scores = scores[keep]

    ious_cache = (np.ones((len(gts), len(preds))) * -1) if (len(gts) and len(preds)) else None

    gts_work = gts.copy()
    tp, fp = 0, 0
    pred_is_tp = np.zeros(len(preds), dtype=bool)

    for predIdx in range(len(preds)):
        best = findBestMatch(gts_work, preds[predIdx], predIdx, iou_threshold, form, ious_cache)
        if best >= 0:
            tp += 1
            pred_is_tp[predIdx] = True
            gts_work[best] = -1
        else:
            fp += 1

    fn = (gts_work.sum(axis=1) > 0).sum()
    return tp, fp, fn, pred_is_tp, (scores if scores is not None else None)


# ----------------------------- Aggregate metrics -----------------------------

def f1_from_counts(tp, fp, fn) -> float:
    """F1 = 2TP / (2TP + FP + FN)."""
    denom = (2 * tp + fp + fn)
    return float((2 * tp) / denom) if denom > 0 else 0.0


def recall_from_counts(tp, fn) -> float:
    """Recall = TP / (TP + FN)."""
    denom = (tp + fn)
    return float(tp / denom) if denom > 0 else 0.0


def precision_from_counts(tp, fp, fn) -> float:
    """Kaggle-style detection precision = TP / (TP + FP + FN)."""
    denom = (tp + fp + fn)
    return float(tp / denom) if denom > 0 else 0.0


def average_precision_from_scores(all_pred_is_tp, all_scores, total_num_gt) -> float:
    """
    PR-AUC (Average Precision) over the whole dataset using confidence-sorted predictions.
    """
    if total_num_gt == 0:
        return 0.0
    all_scores = np.asarray(all_scores)
    if all_scores.size == 0:
        return 0.0

    order = np.argsort(-all_scores)
    tp_cum = fp_cum = 0
    precisions, recalls = [], []

    for idx in order:
        if all_pred_is_tp[idx]:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_num_gt)

    # Monotonic precision envelope (VOC/COCO style)
    precisions = np.asarray(precisions, dtype=float)
    recalls = np.asarray(recalls, dtype=float)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Riemann sum over recall deltas (equivalent to trapezoid with stepwise precision)
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precisions, recalls):
        ap += p * max(0.0, r - prev_r)
        prev_r = r
    return float(ap)


def detection_metrics_over_dataset(
    batch_targets,
    batch_outputs,
    iou_threshold: float = 0.5,
    form: str = 'coco',
    score_threshold: float = 0.5,
    compute_ap: bool = True
):
    """
    Aggregate dataset-level metrics.

    Returns:
        {
          'precision': TP/(TP+FP+FN)  # Kaggle-style
          'recall':    TP/(TP+FN)
          'f1':        2TP/(2TP+FP+FN)
          'ap':        PR-AUC (or None if compute_ap=False)
        }
    """
    def _to_numpy(x):
        # Accept torch.Tensor or array-like; return np.ndarray on CPU.
        try:
            return x.detach().cpu().numpy()
        except AttributeError:
            return np.asarray(x)

    total_tp = total_fp = total_fn = 0
    total_num_gt = 0
    all_pred_is_tp, all_scores = [], []

    for t, o in zip(batch_targets, batch_outputs):
        gts = _to_numpy(t['boxes'])
        preds = _to_numpy(o['boxes'])
        scores = _to_numpy(o['scores'])

        total_num_gt += len(gts)

        tp, fp, fn, pred_is_tp, kept_scores = image_counts_with_score(
            gts, preds, scores,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            form=form
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if compute_ap and kept_scores is not None and len(kept_scores):
            all_pred_is_tp.extend(pred_is_tp.tolist())
            all_scores.extend(kept_scores.tolist())

    precision = precision_from_counts(total_tp, total_fp, total_fn)
    recall = recall_from_counts(total_tp, total_fn)
    f1 = f1_from_counts(total_tp, total_fp, total_fn)
    ap = average_precision_from_scores(np.array(all_pred_is_tp), np.array(all_scores), total_num_gt) if compute_ap else None

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap': (float(ap) if ap is not None else None),
    }

if __name__ == "__main__":
    # One GT, two preds: one good, one bad.
    gts   = np.array([[10,10,20,20]])                 # pascal_voc
    preds = np.array([[11,11,19,19], [100,100,120,120]])
    scores= np.array([0.9, 0.2])

    tp, fp, fn, pred_is_tp, kept_scores = image_counts_with_score(
        gts, preds, scores, score_threshold=0.0, iou_threshold=0.5, form='pascal_voc'
    )
    assert (tp, fp, fn) == (1, 1, 0), (tp, fp, fn)

    metrics = detection_metrics_over_dataset(
        [{'boxes': gts}],
        [{'boxes': preds, 'scores': scores}],
        iou_threshold=0.5, form='pascal_voc', score_threshold=0.0, compute_ap=True
    )
    print(metrics)  # should show precision=1/(1+1+0)=0.5, recall=1.0, f1≈0.666...
