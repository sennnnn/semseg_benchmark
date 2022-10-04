import sys
import random
import os
import os.path as osp

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

sys.path.append('./')

from sswss.core import get_arguments  # noqa
from sswss.apis import evaluate  # noqa
from sswss.datasets import get_num_classes, get_class_names, build_dataset  # noqa
from sswss.core import DistBaseTrainer, cfg, cfg_from_file, cfg_from_list  # noqa
from sswss.utils import (  # noqa
    Checkpointer, Writer, convert_model, is_enabled, is_main_process, reduce_dict, reduce, build_dataloader,
    init_process_group)
from sswss.utils.lr_scheduler import EpochLrScheduler


class DecTrainer(DistBaseTrainer):

    def __init__(self, cfg, model, optim, lr_sche, train_loader, val_loader, writer, checkpointer, **kwargs):
        super(DecTrainer, self).__init__(cfg, model, optim, lr_sche, train_loader, val_loader, writer, checkpointer,
                                         **kwargs)

        self.nclass = get_num_classes(cfg.DATASET.NAME)
        self.classNames = get_class_names(cfg.DATASET.NAME)
        assert self.nclass == len(self.classNames)

        self.start_epoch = self.checkpointer.start_epoch
        self.start_iter = self.checkpointer.start_iter
        self.best_score = self.checkpointer.best_score

        self._iter = self.start_iter
        self._epoch = self.start_epoch

    def train_epoch(self, epoch):
        assert self._epoch == epoch
        self.model.train()

        PRETRAIN = self._epoch <= cfg.TRAIN.PRETRAIN
        self.writer.update_data_timepoint()
        for dataset_dict in self.train_loader:
            self.writer.update_forward_timepoint()
            # to cuda
            for k in ['img', 'pix_gt', 'img_gt', 'raw_img']:
                if isinstance(dataset_dict[k], torch.Tensor):
                    dataset_dict[k] = dataset_dict[k].cuda()

            # forward
            dataset_dict['PRETRAIN'] = PRETRAIN
            output = self.model(dataset_dict)
            losses = {k: v.mean() for k, v in output.items() if k.startswith('loss')}
            loss = sum(losses.values())

            self.writer.update_backward_timepoint()
            # backward
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.writer.update_data_timepoint()
            # log
            loss = reduce(loss)
            losses = reduce_dict(losses)
            if is_main_process():
                log_losses = {k: v.item() for k, v in losses.items() if k.startswith('loss')}
                log_losses['total_loss'] = loss.item()
                self.writer._iter_log_losses(log_losses, self.optim.param_groups[0]['lr'], self._epoch, self._iter)
            self._iter += 1

        self._epoch += 1

    def validation(self, checkpoint=False):
        self.model.eval()
        IoU_dict = {}
        for key in ['cam']:
            with torch.no_grad():
                IoU = evaluate(self.model, self.val_loader, False)
            IoU_dict[key] = IoU

        self.writer._epoch_log_eval(self.classNames, IoU_dict, self._epoch - 1)
        if checkpoint:
            self.checkpointer.checkpoint(self._epoch - 1, self._iter - 1, np.mean(IoU_dict['cam']))

    def train(self):
        for epoch in range(self.start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):
            self.lr_sche.adjust_learning_rate(epoch)

            # shuffle
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch()

            self.train_epoch(epoch)

            with torch.no_grad():
                self.validation(True)

        self.writer.close()


def main_worker(rank, num_gpus, args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    init_process_group('nccl', num_gpus, rank)

    if args.work_dir is None:
        cfg_name = osp.splitext(osp.basename(args.cfg_file))[0]
        dir_name = osp.split(osp.dirname(args.cfg_file))[1]
        cfg.WORK_DIR = osp.join('work_dirs', dir_name, cfg_name)
    else:
        cfg.WORK_DIR = args.work_dir

    if not osp.exists(cfg.WORK_DIR):
        os.makedirs(cfg.WORK_DIR, 0o775)

    # Reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if is_enabled():
        cfg.NET.BN_TYPE = 'syncbn'

    model = DecTrainer.build_model(cfg)
    kwargs = {
        "base_lr": cfg.TRAIN.LR,
        "wd": cfg.TRAIN.WEIGHT_DECAY,
        "batch_size": cfg.TRAIN.BATCH_SIZE,
        "world_size": num_gpus,
    }
    param_groups = model.parameter_groups(**kwargs)
    optim = DecTrainer.build_optim(param_groups, cfg.TRAIN)
    lr_sche = EpochLrScheduler(cfg, optim)

    checkpointer = Checkpointer(cfg.WORK_DIR, max_n=3)
    checkpointer.add_model(model, optim)

    writer = Writer(cfg.WORK_DIR)
    writer._build_writers()

    if args.resume is not None:
        checkpointer.load(args.resume)

    # to ddp model
    model = convert_model(model, find_unused_parameters=True)

    train_dataset = build_dataset(cfg, cfg.DATASET.TRAIN_SPLIT)
    val_dataset = build_dataset(cfg, 'val')

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.TRAIN.NUM_WORKERS)

    val_loader = build_dataloader(
        val_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=cfg.TRAIN.NUM_WORKERS)

    trainer = DecTrainer(cfg, model, optim, lr_sche, train_loader, val_loader, writer, checkpointer)
    if is_main_process():
        trainer.writer.logger_writer.info(model)
        trainer.writer.logger_writer.info("Config: ")
        trainer.writer.logger_writer.info(cfg)
    trainer.train()


def main():
    args = get_arguments(sys.argv[1:])

    num_gpus = args.num_gpus

    if num_gpus > 1:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        mp.spawn(main_worker, nprocs=num_gpus, args=(num_gpus, args))
    else:
        # Simply call main_worker function
        main_worker(0, num_gpus, args)


if __name__ == "__main__":
    main()
