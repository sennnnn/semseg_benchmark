import os
import sys
import numpy as np

import torch
import torch.multiprocessing as mp

sys.path.append('./')

from sswss.core.opts import get_arguments  # noqa
from sswss.core.config import cfg, cfg_from_file, cfg_from_list  # noqa
from sswss.models import build_model  # noqa
from sswss.datasets import build_dataset  # noqa
from sswss.utils import (convert_model, is_enabled, is_main_process, build_dataloader, init_process_group)  # noqa
from sswss.apis import evaluate  # noqa


def main_worker(rank, num_gpus, args):
    init_process_group('nccl', num_gpus, rank)

    # Reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

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
        # model.load_state_dict(torch.load(args.resume, map_location='cpu')['model'])

    # raw_state_dict['model'] = model.state_dict()
    # torch.save(raw_state_dict, 'debug.pth')
    # exit(0)

    # model.load_state_dict(
    #     torch.load('../../reproduces/weak_sup_semseg/MCTformer/MCTformerV2.pth', map_location='cpu')['model'])

    # model.load_state_dict(
    #     torch.load(
    #         '../../reproduces/weak_sup_semseg/MCTformer/work_dirs/MCTformer_v2_1.0/checkpoint_best.pth',
    #         map_location='cpu')['model'])

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
        IoU = evaluate(model, val_loader, cfg.TEST.USE_GT_LABELS, False, cfg.TEST.SCALES,
                       ['none', 'horizontal'] if cfg.TEST.FLIP else ['none'])
        # from sswss.apis.eval import evaluate_wvisualize
        # IoU = evaluate_wvisualize(model, val_loader, cfg.TEST.USE_GT_LABELS, False, cfg.TEST.SCALES,
        #                           ['none', 'horizontal'] if cfg.TEST.FLIP else ['none'])

    if is_main_process():
        print(IoU)
        print(np.mean(IoU))

    # with torch.no_grad():
    #     IoU = evaluate(model, val_loader, cfg.TEST.USE_GT_LABELS, False, cfg.TEST.SCALES,
    #                    ['none', 'horizontal'] if cfg.TEST.FLIP else ['none'], 'pseudo_gt_onehot')

    # if is_main_process():
    #     print(IoU)
    #     print(np.mean(IoU))

    # with torch.no_grad():
    #     IoU = evaluate(model, val_loader, cfg.TEST.USE_GT_LABELS, False, cfg.TEST.SCALES,
    #                    ['none', 'horizontal'] if cfg.TEST.FLIP else ['none'], 'pseudo_slic_gt_onehot')

    # if is_main_process():
    #     print(IoU)
    #     print(np.mean(IoU))


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
