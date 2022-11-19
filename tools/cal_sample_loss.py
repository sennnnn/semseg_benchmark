import argparse
import os
import os.path as osp
import sys
import numpy as np
import pickle

import torch
import torch.multiprocessing as mp

sys.path.append('./')

from sembm.core.config import cfg, cfg_from_file, cfg_from_list  # noqa
from sembm.models import build_model  # noqa
from sembm.datasets import build_dataset  # noqa
from sembm.utils import (convert_model, is_enabled, is_main_process, build_dataloader, init_process_group)  # noqa
from sembm.apis.eval import inference


def parse_args():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model Evaluation")

    parser.add_argument("--resume", type=str, default=None, help="Snapshot \"ID,iter\" to load")

    parser.add_argument('--num-gpus', default=1, type=int, help='number of gpus for distributed training')
    parser.add_argument('--port', default='10002', type=str, help='port used to set up distributed training')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')

    #
    # Configuration
    #
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True, help='Config file for training (and optionally testing)')
    parser.add_argument("--out-path", type=str)

    args = parser.parse_args()

    return args


def main_worker(rank, num_gpus, args):
    init_process_group('nccl', num_gpus, rank)

    # Reading the config
    cfg_from_file(args.cfg_file)

    if is_enabled():
        cfg.NET.BN_TYPE = 'syncbn'

    model = build_model(cfg)
    model.eval()

    if args.resume is not None:
        raw_state_dict = torch.load(args.resume, map_location='cpu')
        state_dict = raw_state_dict
        _state_dict = model.state_dict()
        new_state_dict = {}
        old_keys = list(state_dict.keys())
        new_keys = list(_state_dict.keys())
        i = 0
        j = 0
        while i < len(old_keys) and j < len(new_keys):
            old_k = old_keys[i]
            new_k = new_keys[j]

            if 'num_batches_tracked' in new_k:
                j += 1
                continue

            if 'num_batches_tracked' in old_k:
                i += 1
                continue

            new_state_dict[new_k] = state_dict[old_k]
            i += 1
            j += 1

        model.load_state_dict(new_state_dict)

    # to ddp model
    model = convert_model(model, find_unused_parameters=True)

    val_dataset = build_dataset(cfg, 'val')

    val_loader = build_dataloader(
        val_dataset,
        batch_size=num_gpus,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg.TRAIN.NUM_WORKERS)

    # with torch.cuda.amp.autocast():
    with torch.no_grad():
        result = inference(model, val_loader, False, cfg.TEST.SCALES,
                           ['none', 'horizontal'] if cfg.TEST.FLIP else ['none'])

    if is_main_process():
        dir_name = osp.dirname(args.out_path)
        if not osp.exists(dir_name):
            os.makedirs(dir_name, 0o775)
        pickle.dump(result, open(args.out_path, 'wb'))


def main():
    args = parse_args()

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
