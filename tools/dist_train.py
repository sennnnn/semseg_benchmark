import sys
import random
import os
import os.path as osp

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

sys.path.append('./')

from sembm.core import get_arguments  # noqa
from sembm.apis import evaluate  # noqa
from sembm.datasets import get_num_classes, get_class_names, build_dataset  # noqa
from sembm.core import DistBaseTrainer, cfg, cfg_from_file, cfg_from_list  # noqa
from sembm.utils import (  # noqa
    Checkpointer, Writer, convert_model, is_enabled, is_main_process, reduce_dict, reduce, build_dataloader,
    init_process_group, LrScheduler)


class IterLoader:

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


class Trainer(DistBaseTrainer):

    def __init__(self, cfg, model, optim, lr_sche, train_loader, val_loader, writer, checkpointer, **kwargs):
        super(Trainer, self).__init__(cfg, model, optim, lr_sche, train_loader, val_loader, writer, checkpointer,
                                      **kwargs)

        self.nclass = get_num_classes(cfg.DATASET.NAME)
        self.classNames = get_class_names(cfg.DATASET.NAME)
        assert self.nclass == len(self.classNames)

        self.start_epoch = self.checkpointer.start_epoch
        self.start_iter = self.checkpointer.start_iter
        self.best_score = self.checkpointer.best_score

        self.eval_period = cfg.TRAIN.EVAL_PERIOD
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
            for k in dataset_dict.keys():
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

            self.lr_sche.adjust_learning_rate(self._epoch, self._iter)

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

    def train_iter(self, iter, iter_loader):
        assert self._iter == iter
        self.model.train()

        PRETRAIN = self._epoch <= cfg.TRAIN.PRETRAIN
        self.writer.update_data_timepoint()
        dataset_dict = next(iter_loader)
        self.writer.update_forward_timepoint()
        # to cuda
        for k in dataset_dict.keys():
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

        self.lr_sche.adjust_learning_rate(self._epoch, self._iter)

        self.writer.update_data_timepoint()
        # log
        loss = reduce(loss)
        losses = reduce_dict(losses)
        if is_main_process():
            log_losses = {k: v.item() for k, v in losses.items() if k.startswith('loss')}
            log_losses['total_loss'] = loss.item()
            self.writer._iter_log_losses(log_losses, self.optim.param_groups[0]['lr'], self._epoch, self._iter)

        self._epoch = iter_loader.epoch + 1
        self._iter += 1

    def validation(self, epoch, iter, checkpoint=False):
        self.model.eval()
        IoU_dict = {}
        for key in ['cam']:
            with torch.no_grad():
                IoU = evaluate(self.model, self.val_loader, False)
            IoU_dict[key] = IoU

        self.writer._log_eval(self.classNames, IoU_dict, epoch, iter)
        if checkpoint:
            self.checkpointer.checkpoint(epoch, iter, np.mean(IoU_dict['cam']))

    def train_epochwise(self):
        for epoch in range(self.start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):
            # shuffle
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            self.train_epoch(epoch)

            with torch.no_grad():
                self.validation(epoch, self._iter - 1, True)

        self.writer.close()

    def train_iterwise(self):
        iter_loader = IterLoader(self.train_loader)
        for iter in range(self.start_iter, cfg.TRAIN.NUM_ITERS + 1):
            self.train_iter(iter, iter_loader)

            if iter % self.eval_period == 0:
                with torch.no_grad():
                    self.validation(self._epoch, iter, True)


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

    if is_main_process():
        if not osp.exists(cfg.WORK_DIR):
            os.makedirs(cfg.WORK_DIR, 0o775)

    # Reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    train_dataset = build_dataset(cfg, cfg.DATASET.TRAIN_SPLIT, test_mode=False)
    val_dataset = build_dataset(cfg, 'val', test_mode=True)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.TRAIN.NUM_WORKERS)

    val_loader = build_dataloader(
        val_dataset,
        batch_size=num_gpus,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg.TRAIN.NUM_WORKERS)

    if is_enabled():
        cfg.NET.BN_TYPE = 'syncbn'

    model = Trainer.build_model(cfg)
    kwargs = {
        "base_lr": cfg.TRAIN.LR,
        "wd": cfg.TRAIN.WEIGHT_DECAY,
        "batch_size": cfg.TRAIN.BATCH_SIZE,
        "world_size": num_gpus,
    }
    param_groups = model.parameter_groups(**kwargs)
    optim = Trainer.build_optim(param_groups, cfg.TRAIN)
    max_epochs = cfg.TRAIN.NUM_EPOCHS
    max_iters = cfg.TRAIN.NUM_EPOCHS * len(train_dataset) // cfg.TRAIN.BATCH_SIZE
    lr_sche = LrScheduler(cfg, max_epochs, max_iters, optim)

    checkpointer = Checkpointer(cfg.WORK_DIR, max_n=3)
    checkpointer.add_model(model, optim)

    writer = Writer(cfg.WORK_DIR)
    writer._build_writers()

    if args.resume is not None:
        checkpointer.load(args.resume)

    # to ddp model
    model = convert_model(model, find_unused_parameters=True)

    trainer = Trainer(cfg, model, optim, lr_sche, train_loader, val_loader, writer, checkpointer)
    if is_main_process():
        trainer.writer.logger_writer.info(model)
        trainer.writer.logger_writer.info("Config: ")
        trainer.writer.logger_writer.info(cfg)
    if cfg.TRAIN.MODE == 'iter':
        trainer.train_iterwise()
    elif cfg.TRAIN.MODE == 'epoch':
        trainer.train_epochwise()


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
