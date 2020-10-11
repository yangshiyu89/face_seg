import math
class LRScheduler(object):
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0, lr_step=0, warmup_epochs=0):
        self.mode = mode
        print("Using {} learning rate scheduler!".format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = iters_per_epoch * num_epochs
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplementedError
        # warmup scheduler
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=> epochs {},\tlearning rate {:.4f},\t\
                   previous best {:.4f}'.format(epoch, lr, best_pred))
        self.epoch = epoch
        assert lr >=0
        self._adjust_lr(lr, optimizer)

    def _adjust_lr(self, lr, optimizer):
        optimizer.param_groups[0]['lr'] = lr

if __name__ == "__main__":

    import torch
    import sys
    sys.path.append('./')
    import models.unet as unet
    model= unet.Unet()
    model.cuda()
    mode = 'poly'
    base_lr = 0.1
    num_epochs = 100

    training_params = [
        {'params': model.parameters(), 'lr': base_lr, 'momentum': 0.9}
    ]
    optimizer = torch.optim.SGD(training_params)
    i = 3
    epoch = 5
    best_pred = 0.87
    lrsche = LRScheduler(mode, base_lr, num_epochs, iters_per_epoch=20)
    lrsche(optimizer, i, epoch, best_pred)



