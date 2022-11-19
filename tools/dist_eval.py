import argparse
import os
import sys
import numpy as np

import torch
import torch.multiprocessing as mp

sys.path.append('./')

from sembm.core.config import cfg, cfg_from_file, cfg_from_list  # noqa
from sembm.models import build_model  # noqa
from sembm.datasets import build_dataset  # noqa
from sembm.utils import (convert_model, is_enabled, is_main_process, build_dataloader, init_process_group)  # noqa
from sembm.apis import evaluate  # noqa
from sembm.apis.eval import evaluate_debug, evaluate_wvisualize, inference


def get_arguments(args_in):
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model Evaluation")

    parser.add_argument("--dataset", type=str, help="Determines dataloader to use (only Pascal VOC supported)")
    parser.add_argument("--resume", type=str, default=None, help="Snapshot \"ID,iter\" to load")
    parser.add_argument('--work-dir', type=str, help='workspace for single config file.')

    parser.add_argument('--num-gpus', default=1, type=int, help='number of gpus for distributed training')
    parser.add_argument('--port', default='10002', type=str, help='port used to set up distributed training')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True, help='Config file for training (and optionally testing)')

    parser.add_argument('--show_folder', type=str, help='If starting generation mode and setting generation folder.')
    parser.add_argument(
        '--infer_folder', type=str, help='generate inference results for benchmarking on evaluation server.')

    args = parser.parse_args(args_in)

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
        # raw_state_dict = torch.load(args.resume, map_location='cpu')
        # state_dict = raw_state_dict
        # _state_dict = model.state_dict()
        # new_state_dict = {}
        # old_keys = list(state_dict.keys())
        # new_keys = list(_state_dict.keys())
        # i = 0
        # j = 0
        # while i < len(old_keys) and j < len(new_keys):
        #     old_k = old_keys[i]
        #     new_k = new_keys[j]

        #     if 'num_batches_tracked' in new_k:
        #         j += 1
        #         continue

        #     if 'num_batches_tracked' in old_k:
        #         i += 1
        #         continue

        #     new_state_dict[new_k] = state_dict[old_k]
        #     i += 1
        #     j += 1

        # model.load_state_dict(new_state_dict)
        model.load_state_dict(torch.load(args.resume, map_location='cpu')['model'])

    # to ddp model
    model = convert_model(model, find_unused_parameters=True)

    eval_dataset = build_dataset(cfg, cfg.DATASET.TEST_SPLIT, test_mode=True)

    eval_loader = build_dataloader(
        eval_dataset,
        batch_size=num_gpus,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg.TRAIN.NUM_WORKERS)

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if args.infer_folder is not None:
                inference(model, eval_loader, cfg.TEST.CRF, cfg.TEST.SCALES,
                          ['none', 'horizontal'] if cfg.TEST.FLIP else ['none'], args.infer_folder)
                return
            if args.show_folder is not None:
                IoU = evaluate_wvisualize(model, eval_loader, cfg.TEST.USE_GT_LABELS, cfg.TEST.CRF, cfg.TEST.SCALES,
                                          ['none', 'horizontal'] if cfg.TEST.FLIP else ['none'], args.show_folder)
            else:
                IoU = evaluate(model, eval_loader, cfg.TEST.USE_GT_LABELS, cfg.TEST.CRF, cfg.TEST.SCALES,
                               ['none', 'horizontal'] if cfg.TEST.FLIP else ['none'])

    if is_main_process():
        print(IoU)
        print(np.mean(IoU))


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
