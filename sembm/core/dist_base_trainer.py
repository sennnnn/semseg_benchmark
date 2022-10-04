from torch import optim

from ..datasets.pascal_voc import load_pascal_voc
from ..models import build_model


class DistBaseTrainer(object):

    def __init__(self, cfg, model, optim, lr_sche, train_loader, val_loader, writer, checkpointer):
        self.cfg = cfg
        self.model = model
        self.optim = optim
        self.lr_sche = lr_sche
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.checkpointer = checkpointer
        self.work_dir = cfg.WORK_DIR

    @staticmethod
    def build_loader(cfg):
        if cfg.DATASET.NAME == 'pascal_voc':
            train_loader, val_loader = load_pascal_voc(cfg, cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.NUM_WORKERS)

        return train_loader, val_loader

    @staticmethod
    def build_model(cfg):
        model = build_model(cfg).cuda()
        # model = get_model(cfg).cuda()

        return model

    @staticmethod
    def build_optim(params, cfg):

        if cfg.OPT == 'Adam':
            upd = optim.Adam(params, lr=cfg.LR, betas=(cfg.BETA1, 0.999), weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPT == 'Adamw':
            upd = optim.AdamW(params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPT == 'SGD':
            print("Using SGD >>> learning rate = {:4.3e}, momentum = {:4.3e}, weight decay = {:4.3e}".format(
                cfg.LR, cfg.MOMENTUM, cfg.WEIGHT_DECAY))
            upd = optim.SGD(params, lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        else:
            print("Optimiser {} not supported".format(cfg.OPT))
            raise NotImplementedError

        upd.zero_grad()

        return upd

    @staticmethod
    def set_lr(optim, lr):
        for param_group in optim.param_groups:
            param_group['lr'] = lr
