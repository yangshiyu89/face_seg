import torch
import torch.nn as nn
class SegLoss(object):
    def __init__(self, weight=None,  ignore_index=255, reduction='mean', batch_average=True):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.batch_average = batch_average

    def build_loss(self, mode='ce'):
        if mode == 'ce':
            return self.crossentropy
        elif mode == 'focal':
            return self.focalloss
        else:
            return NotImplementedError

    def crossentropy(self, logit, target):
        n, _, _, _, = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        criterion.cuda()
        loss = criterion(logit, target)
        if self.batch_average:
            loss /= n
        return loss

    def focalloss(self, logit, target, gamma=2, alpha=0.5):
        n, _, _, _, = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        criterion.cuda()
        logpt = criterion(logit, target)
        pt = torch.exp(-logpt)
        print(logpt, pt)
        loss = ((1 - pt) ** gamma )* logpt
        if alpha:
            loss *= alpha
        if self.batch_average:
            loss /= n
        return loss
if __name__ == "__main__":
    predicts = torch.rand(2, 8, 480, 480).cuda()
    targets = torch.rand(2, 480, 480).long().cuda()
    loss = SegLoss().focalloss(predicts, targets)
    print(loss.item())

