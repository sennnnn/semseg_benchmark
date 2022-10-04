import os
import os.path as osp
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append('./')

from sswss.core.opts import get_arguments
from sswss.core.config import cfg, cfg_from_file, cfg_from_list
from sswss.models import build_model
from sswss.datasets import build_dataset
from sswss.utils.distributed import build_dataloader

from sswss.utils.dcrf import crf_inference
from sswss.utils.misc import histc


def calculate_iou(pred, gt, num_classes):
    from sswss.utils.misc import histc
    inter = histc(pred[gt == pred].float(), bins=(num_classes), min=0, max=num_classes - 1)
    pred_area = histc(pred.float(), bins=(num_classes), min=0, max=num_classes - 1)
    gt_area = histc(gt.float(), bins=(num_classes), min=0, max=num_classes - 1)

    union = pred_area + gt_area - inter

    return inter / union


def augmentation(img, scale, flip_direction, raw_size, size_divisor=None):
    H, W = raw_size
    tH, tW = int(round(H * scale)), int(round(W * scale))
    # Augmentation
    img = F.interpolate(img, (tH, tW), mode='bilinear', align_corners=False)

    if flip_direction == 'horizontal':
        img = torch.flip(img, dims=[-1])
    if flip_direction == 'vertical':
        img = torch.flip(img, dims=[-2])
    if flip_direction == 'diagonal':
        img = torch.flip(img, dims=[-2, -1])

    if size_divisor is not None:
        sd = size_divisor
        pH, pW = (tH + sd) // sd * sd, (tW + sd) // sd * sd
        padH, padW = pH - tH, pW - tW
        img = F.pad(img, (0, padW, 0, padH), value=0, mode='constant')

    return img


def reverse_augmentation(img, scale, flip_direction, raw_size, size_divisor=None):
    # Reverse Augmentation
    H, W = raw_size
    if size_divisor is not None:
        tH, tW = int(round(H * scale)), int(round(W * scale))
        img = img[:, :, :tH, :tW]

    if flip_direction == 'horizontal':
        img = torch.flip(img, dims=[-1])
    if flip_direction == 'vertical':
        img = torch.flip(img, dims=[-2])
    if flip_direction == 'diagonal':
        img = torch.flip(img, dims=[-2, -1])

    img = F.interpolate(img, (H, W), mode='bilinear', align_corners=False)

    return img


def tta_inference(model,
                  raw_img,
                  img,
                  img_gt,
                  pix_gt,
                  scales=[1.0],
                  flip_directions=['none'],
                  output_key='pix_probs_seg'):
    H, W = raw_img.shape[-2:]
    img_preds = []
    pix_preds = []
    for scale in scales:
        for flip_direction in flip_directions:
            simg = augmentation(img, scale, flip_direction, (H, W))

            batched_input = {}
            batched_input['img'] = simg
            batched_input['raw_img'] = raw_img
            batched_input['img_gt'] = img_gt
            batched_input['pix_gt'] = pix_gt

            outputs = model(batched_input)

            img_pred = outputs['img_logits']
            pix_pred = outputs[output_key]

            pix_pred = reverse_augmentation(pix_pred, scale, flip_direction, (H, W))

            for k, v in outputs.items():
                Cc = img_pred.shape[1]
                Cs = pix_pred.shape[1]
                if img_gt is not None:
                    pix_pred[:, (Cs - Cc):] = pix_pred[:, (Cs - Cc):] * img_gt[:, :, None, None].float()
                else:
                    img_sigmoid = torch.sigmoid(img_pred)
                    img_gt = (img_sigmoid > 0.3)
                    pix_pred[:, (Cs - Cc):] = pix_pred[:, (Cs - Cc):] * img_gt[:, :, None, None].float()

            img_preds.append(img_pred)
            pix_preds.append(pix_pred)

    img_pred = sum(img_preds) / len(img_preds)
    pix_pred = sum(pix_preds) / len(pix_preds)

    return img_pred, pix_pred


def evaluate(model,
             loader,
             gt_label_filter=False,
             crf=False,
             scales=[1.0],
             flip_directions=['none'],
             output_key='pix_probs_cam',
             show_folder=None):
    inter_list = []
    union_list = []

    num_classes = loader.dataset.NUM_CLASSES

    for n, dataset_dict in enumerate(tqdm(loader)):
        raw_img = dataset_dict['raw_img'].cuda()
        img = dataset_dict['img'].cuda()
        pix_gt = dataset_dict['pix_gt'].cuda()
        img_gt = dataset_dict['img_gt'].cuda()
        img_pred, pix_pred = tta_inference(model, raw_img, img, img_gt if gt_label_filter else None, pix_gt, scales,
                                           flip_directions, output_key)
        pix_pred = pix_pred[0]
        pix_gt = pix_gt[0]
        pix_pred[0, ::] = torch.pow(pix_pred[0, ::], 3)
        pred = torch.argmax(pix_pred, dim=0)
        if crf:
            raw_img = raw_img[0].permute(1, 2, 0).contiguous()
            pred_crf = crf_inference(
                raw_img.cpu().numpy().astype(np.uint8), pix_pred.cpu().numpy(), t=10, scale_factor=1, labels=21)
            pred_crf = torch.argmax(torch.tensor(pred_crf, device=pred.device), dim=0)
            pred = pred_crf

        gt = pix_gt.long()
        class_ids = torch.nonzero(img_gt[0])[:, 0].cpu().numpy().tolist()
        classes = [loader.dataset.CLASSES[idx + 1] for idx in class_ids]

        nounlabeled_pred = pred[gt != 255]
        nounlabeled_gt = gt[gt != 255]

        inter = histc(
            nounlabeled_pred[nounlabeled_gt == nounlabeled_pred].float(), bins=num_classes, min=0, max=num_classes - 1)
        pred_area = histc(nounlabeled_pred.float(), bins=num_classes, min=0, max=num_classes - 1)
        gt_area = histc(nounlabeled_gt.float(), bins=num_classes, min=0, max=num_classes - 1)
        union = pred_area + gt_area - inter

        if show_folder is not None:
            if not osp.exists(show_folder):
                os.makedirs(show_folder, 0o775)
            filename = osp.splitext(dataset_dict['filename'][0])[0]
            show_img = Image.fromarray(raw_img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            # show_img.save(osp.join(show_folder, filename + '_img.png'))
            show_gt = Image.fromarray(gt.cpu().numpy().astype(np.uint8)).convert('L')
            show_gt.putpalette(loader.dataset.get_palette())
            # show_gt.save(osp.join(show_folder, filename + '_gt.png'))
            show_pred = Image.fromarray(pred.cpu().numpy().astype(np.uint8)).convert('L')
            show_pred.putpalette(loader.dataset.get_palette())
            # show_pred.save(osp.join(show_folder, filename + '_pred.png'))
            ROW = 1
            COL = 3
            UNIT_WIDTH_SIZE = show_img.width
            UNIT_HEIGHT_SIZE = show_img.height
            target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL + 5 * (COL - 1), UNIT_HEIGHT_SIZE * ROW), (255, 255, 255))
            target.paste(show_img, (UNIT_WIDTH_SIZE * 0, UNIT_HEIGHT_SIZE * 0))
            target.paste(show_gt, (UNIT_WIDTH_SIZE * 1 + 5, UNIT_HEIGHT_SIZE * 0))
            target.paste(show_pred, (UNIT_WIDTH_SIZE * 2 + 10, UNIT_HEIGHT_SIZE * 0))
            target.save(osp.join(show_folder, filename + '.png'))
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.subplot(131)
            plt.imshow(np.array(show_img))
            plt.axis('off')
            plt.title('raw_img')
            plt.subplot(132)
            plt.imshow(np.array(show_gt.convert('RGB')))
            plt.axis('off')
            plt.title('gt')
            plt.subplot(133)
            plt.imshow(np.array(show_pred.convert('RGB')))
            plt.axis('off')
            iou = inter / (union + 1e-6)
            bg_iou = iou[0]
            fg_iou = torch.mean(iou[img_gt[0].cpu() == 1])
            plt.title(f'w/o bg: bg_iou {int(bg_iou.cpu().numpy()*100)} fg_iou {int(fg_iou.cpu().numpy()*100)}')
            plt.suptitle(' '.join(classes))
            plt.tight_layout()
            plt.savefig(osp.join(show_folder, filename + '_show.png'))
            plt.close()

        inter_list.append(inter)
        union_list.append(union)

    IoU = sum(inter_list) / sum(union_list)
    return IoU.cpu().numpy()


if __name__ == '__main__':

    # loading the model
    args = get_arguments(sys.argv[1:])

    # reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # Loading the model
    model = build_model(cfg)
    if args.resume is not None:
        # state_dict = torch.load(args.resume)['model']
        # _state_dict = model.state_dict()
        # new_state_dict = {}
        # old_keys = list(state_dict.keys())
        # new_keys = list(_state_dict.keys())
        # i = 0
        # j = 0
        # while i < len(old_keys) and j < len(new_keys):
        #     old_k = old_keys[i]
        #     new_k = new_keys[j]
        #     continue_flag = 0
        #     if 'num_batches_tracked' in old_k:
        #         i += 1
        #         continue_flag = 1
        #     if 'num_batches_tracked' in new_k:
        #         j += 1
        #         continue_flag = 1
        #     if continue_flag:
        #         continue
        #     new_state_dict[new_k] = state_dict[old_k]
        #     i += 1
        #     j += 1

        # model.load_state_dict(new_state_dict)
        model.load_state_dict(torch.load(args.resume)['model'])
    # checkpoint = Checkpoint(args.snapshot_dir, max_n=5)
    # checkpoint.add_model('enc', model)
    # checkpoint.load(args.resume)
    model.eval()
    model = model.cuda()

    loader = build_dataloader(build_dataset(cfg, 'val'), batch_size=1, num_workers=cfg.TRAIN.NUM_WORKERS)

    with torch.no_grad():
        IoU = evaluate(
            model,
            loader,
            cfg.TEST.USE_GT_LABELS,
            False,
            cfg.TEST.SCALES, ['none', 'horizontal'] if cfg.TEST.FLIP else ['none'],
            show_folder=None)

    print(IoU)
    print(np.mean(IoU))
