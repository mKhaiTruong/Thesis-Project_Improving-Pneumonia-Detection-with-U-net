import torch.nn as nn
import torch
import segmentation_models_pytorch as smp

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