import os.path as osp
import time

import numpy as np
from prettytable import PrettyTable

try:  # backward compatibility
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

from sswss.utils.logging import get_logger
from sswss.utils.distributed import is_main_process


class Writer(object):

    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.data_timepoints = []
        self.forward_timepoints = []
        self.backward_timepoints = []

    def update_data_timepoint(self):
        self.data_timepoints.append(time.time())

    def update_forward_timepoint(self):
        self.forward_timepoints.append(time.time())

    def update_backward_timepoint(self):
        self.backward_timepoints.append(time.time())

    def _build_writers(self):
        if is_main_process():
            self.tensorboard_writer = SummaryWriter(osp.join(self.work_dir, 'tensorboard'))
            self.logger_writer = get_logger('sswss', osp.join(self.work_dir, 'record.log'))

    def _iter_log_losses(self, losses, lr, epoch, iter):
        if is_main_process():
            msg = "Loss [{:03d}] | [{:07d}]: ".format(epoch, iter)
            for k, v in losses.items():
                self.tensorboard_writer.add_scalar(f'Loss/{k}', v, iter)
                msg += "{}: {:.4f} | ".format(k, v)
            msg += f"LR: {lr:.5f} | "
            msg += f"data time: {self.forward_timepoints[-1] - self.data_timepoints[-2]:.4f}s | "
            msg += f"forward time: {self.backward_timepoints[-1] - self.forward_timepoints[-1]:.4f}s | "
            msg += f"backward time: {self.data_timepoints[-1] - self.backward_timepoints[-1]:.4f}s | "


            if iter % 10 == 0:
                self.logger_writer.info(msg)

    def _epoch_log_eval(self, classnames, IoU_dict, epoch):
        if is_main_process():
            class_table = PrettyTable()
            class_table.add_column('Class', classnames + ['Mean'])
            for k, IoU in IoU_dict.items():
                mIoU = np.mean(IoU)
                IoU = np.concatenate([IoU, [mIoU]])
                IoU = np.round(IoU * 100.0, 2)
                class_table.add_column(f'IoU_{k}', IoU)

                self.tensorboard_writer.add_scalar(f'val_mIoU_{k}', mIoU, epoch)
                for idx, class_name in enumerate(classnames):
                    self.tensorboard_writer.add_scalar(f'val_IoU_{k}/{idx:02d}_{class_name}', IoU[idx], epoch)

            log_str = '\n' + class_table.get_string()
            self.logger_writer.info(log_str)

    def close(self):
        if is_main_process():
            self.tensorboard_writer.close()
