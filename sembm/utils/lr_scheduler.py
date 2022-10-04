import math
import numpy as np


class EpochLrScheduler:

    def __init__(self, cfg, max_epochs, max_iters, optimizer):
        self.lr = cfg.TRAIN.LR
        self.lr_scheduler_name = cfg.TRAIN.LR_SCHEDULER_NAME
        self.lr_decay_epochs = cfg.TRAIN.LR_DECAY_EPOCHS
        self.lr_decay_rate = cfg.TRAIN.LR_DECAY_RATE
        self.lr_decay_alpha = cfg.TRAIN.LR_DECAY_ALPHA
        self.max_epochs = max_epochs
        self.max_iters = max_iters

        self.optimizer = optimizer

        self.initial_lrs = [param_group['lr'] for param_group in optimizer.param_groups]

    def adjust_learning_rate(self, epoch, iter):
        for lr, param_group in zip(self.initial_lrs, self.optimizer.param_groups):
            if self.lr_scheduler_name == 'cosine':
                eta_min = lr * (self.lr_decay_rate**3)
                lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / self.max_epochs)) / 2
            elif self.lr_scheduler_name == 'step':
                steps = np.sum(epoch > np.asarray(self.lr_decay_epochs))
                if steps > 0:
                    lr = lr * (self.lr_decay_rate**steps)
            elif self.lr_scheduler_name == 'poly':
                lr_mult = (1 - iter / self.max_iters) ** self.lr_decay_alpha
                lr = lr * lr_mult
            else:
                raise NotImplementedError("Only support step/cosine lr scheduler for epoch learning rate adjustment.")

            param_group['lr'] = lr

# def epoch_adjust_learning_rate(cfg, optimizer, epoch):
#     lr_scheduler_name = cfg.TRAIN.LR_SCHEDULER_NAME
#     lr_decay_epochs = cfg.TRAIN.LR_DECAY_EPOCHS
#     lr_decay_rate = cfg.TRAIN.LR_DECAY_RATE
#     max_epochs = cfg.TRAIN.NUM_EPOCHS

#     for param_group in optimizer.param_groups:
#         lr = param_group['lr']

#         if lr_scheduler_name == 'cosine':
#             eta_min = lr * (lr_decay_rate**3)
#             lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / max_epochs)) / 2
#         elif lr_scheduler_name == 'step':
#             steps = np.sum(epoch > np.asarray(lr_decay_epochs))
#             if steps > 0:
#                 lr = lr * (lr_decay_rate**steps)
#         else:
#             raise NotImplementedError("Only support step/cosine lr scheduler for epoch learning rate adjustment.")

#         param_group['lr'] = lr
