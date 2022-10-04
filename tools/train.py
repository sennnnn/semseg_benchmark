from __future__ import print_function

import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append('./')

from sswss.core.opts import get_arguments
from sswss.datasets import get_dataloader, get_num_classes, get_class_names
from sswss.models import build_model
from sswss.core.base_trainer import BaseTrainer
from sswss.core.config import cfg, cfg_from_file, cfg_from_list
from sswss.utils.timer import Timer
from sswss.apis.eval import evaluate


class DecTrainer(BaseTrainer):

    def __init__(self, args, **kwargs):
        super(DecTrainer, self).__init__(args, **kwargs)

        # dataloader
        self.trainloader = get_dataloader(
            cfg.DATASET.NAME,
            cfg,
            'train',
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            test_mode=False)
        self.valloader = get_dataloader(
            cfg.DATASET.NAME, cfg, 'val', batch_size=1, num_workers=cfg.TRAIN.NUM_WORKERS, test_mode=True)

        self.nclass = get_num_classes(cfg.DATASET.NAME)
        self.classNames = get_class_names(cfg.DATASET.NAME)
        assert self.nclass == len(self.classNames)

        # model
        self.enc = build_model(cfg).cuda()

        # optimizer using different LR
        enc_params = self.enc.parameter_groups(cfg.NET.LR, cfg.NET.WEIGHT_DECAY)
        self.optim_enc = self.get_optim(enc_params, cfg.NET)

        # checkpoint management
        self._define_checkpoint('enc', self.enc, self.optim_enc)
        self._load_checkpoint(args.resume)

        # using cuda
        self.enc = nn.DataParallel(self.enc)

        self._iter = 0

    def train_step(self, epoch, batched_inputs):

        PRETRAIN = epoch < cfg.TRAIN.PRETRAIN

        for k in ['img', 'pix_gt', 'img_gt', 'raw_img']:
            if isinstance(batched_inputs[k], torch.Tensor):
                batched_inputs[k] = batched_inputs[k].cuda()
        batched_inputs['PRETRAIN'] = PRETRAIN
        # classification
        output = self.enc(batched_inputs)
        losses = {k: v.mean() for k, v in output.items() if k.startswith('loss')}
        loss = sum(losses.values())

        if self.enc.training:
            self.optim_enc.zero_grad()
            loss.backward()
            self.optim_enc.step()

        self._iter += 1

        losses["loss"] = loss
        losses = {k: v.item() for k, v in losses.items() if k.startswith('loss')}

        # make sure to cut the return values from graph
        return losses

    def train_epoch(self, epoch):
        self.enc.train()

        # adding stats for classes
        timer = Timer("New Epoch: ")
        for i, dataset_dict in enumerate(self.trainloader):
            losses = self.train_step(epoch, dataset_dict)

            # intermediate logging
            if i % 10 == 0:
                msg = "Loss [{:04d}]: ".format(i)
                for loss_key, loss_val in losses.items():
                    msg += "{}: {:.4f} | ".format(loss_key, loss_val)

                msg += " | Im/Sec: {:.1f}".format(i * self.trainloader.batch_size / timer.get_stage_elapsed())
                self.logger.info(msg)
                sys.stdout.flush()

        # plotting learning rate
        for ii, l in enumerate(self.optim_enc.param_groups):
            self.logger.info("Learning rate [{}]: {:4.3e}".format(ii, l['lr']))
            self.writer.add_scalar('lr/enc_group_%02d' % ii, l['lr'], epoch)

    def validation(self, epoch, checkpoint=False):
        self.enc.eval()
        with torch.no_grad():
            IoU = evaluate(self.enc, self.valloader, False)
        mIoU = np.mean(IoU)

        self.logger.info(f'IoU: {IoU}')
        self.logger.info(f'mIoU: {mIoU}')

        if checkpoint:
            self.checkpoint_best(mIoU, epoch)

        self.writer.add_scalar('val_mIoU', mIoU, epoch)
        for idx, class_name in enumerate(self.classNames):
            self.writer.add_scalar(f'val_IoU/{idx:02d}_{class_name}', IoU[idx], epoch)


if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])

    # Reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    trainer = DecTrainer(args)

    trainer.logger.info("Config: \n")
    trainer.logger.info(cfg)

    timer = Timer()

    def time_call(func, msg, *args, **kwargs):
        timer.reset_stage()
        func(*args, **kwargs)
        trainer.logger.info(msg + (" {:3.2}m".format(timer.get_stage_elapsed() / 60.)))

    for epoch in range(trainer.start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):
        trainer.logger.info("Epoch >>> {}".format(epoch))

        time_call(trainer.train_epoch, "Train epoch: ", epoch)

        if epoch != 0:
            with torch.no_grad():
                time_call(trainer.validation, "Validation /   Val: ", epoch, checkpoint=True)
