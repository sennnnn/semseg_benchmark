import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from skimage.segmentation import slic, mark_boundaries

from ..utils.dcrf import crf_inference
from ..utils.misc import histc
from ..utils.distributed import all_reduce


def calculate_iou(pred, gt, num_classes):
    from ..utils.misc import histc
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
                  output_key='pix_probs_cam'):
    H, W = raw_img.shape[-2:]
    img_preds = []
    pix_preds = []
    for scale in scales:
        for flip_direction in flip_directions:
            simg = augmentation(img, scale, flip_direction, (H, W))
            rimg = augmentation(raw_img.float(), scale, flip_direction, (H, W))

            batched_input = {}
            batched_input['img'] = simg
            batched_input['raw_img'] = rimg
            batched_input['img_gt'] = img_gt
            batched_input['pix_gt'] = pix_gt

            outputs = model(batched_input)

            img_pred = outputs['img_logits']
            pix_pred = outputs[output_key]

            pix_pred = reverse_augmentation(pix_pred, scale, flip_direction, (H, W))

            Cc = img_pred.shape[1]
            Cs = pix_pred.shape[1]
            if img_gt is not None:
                pix_pred[:, (Cs - Cc):] = pix_pred[:, (Cs - Cc):] * img_gt[:, :, None, None]
            else:
                img_sigmoid = torch.sigmoid(img_pred)
                img_gt = (img_sigmoid > 0.3)
                pix_pred[:, (Cs - Cc):] = pix_pred[:, (Cs - Cc):] * img_gt[:, :, None, None]

            img_preds.append(img_pred)
            pix_preds.append(pix_pred)

    img_pred = sum(img_preds) / len(img_preds)
    pix_pred = sum(pix_preds)

    return img_pred, pix_pred


def _crf(pix_pred, raw_img, img_gt):
    orig_img = raw_img[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
    canvas = torch.zeros_like(pix_pred)
    img_gt = img_gt[0]
    ids = [0]
    for cls_id, val in enumerate(img_gt.long()):
        if val == 1:
            ids.append(cls_id + 1)

    pix_pred = pix_pred[ids]
    pix_pred = crf_inference(orig_img, pix_pred.cpu().numpy(), labels=pix_pred.shape[0])
    pix_pred = torch.tensor(pix_pred).cuda()
    canvas[ids] = pix_pred
    pix_pred = canvas

    return pix_pred


def evaluate(model, loader, gt_label_filter=False, crf=False, scales=[1.0], flip_directions=['none']):
    num_classes = loader.dataset.NUM_CLASSES
    total_inter = torch.zeros((num_classes, ))
    total_union = torch.zeros((num_classes, ))

    for n, dataset_dict in enumerate(tqdm(loader)):

        pix_gt = dataset_dict['pix_gt'].cuda()
        dataset_dict['gt_label_filter'] = gt_label_filter
        dataset_dict['scales'] = scales
        dataset_dict['flip_directions'] = flip_directions

        pix_pred = model(dataset_dict)
        pix_pred = pix_pred[0]
        pix_gt = pix_gt[0]

        if crf:
            raw_img = dataset_dict['raw_img'].cuda()
            raw_img = raw_img[0].permute(1, 2, 0).contiguous()
            pix_pred_crf = crf_inference(
                raw_img.cpu().numpy().astype(np.uint8), pix_pred.cpu().numpy(), t=10, scale_factor=1, labels=21)
            pix_pred_crf = torch.tensor(pix_pred_crf, device=pix_pred.device)
            pix_pred = pix_pred_crf

        pred = torch.argmax(pix_pred, dim=0)

        # name = dataset_dict['filename']
        # pred = np.array(
        #     Image.open(
        #         osp.join('../MCTformer/work_dirs/MCTformer_v2_official/pgt-psa-rw', name[0].replace('.jpg', '.png'))))
        # pred = torch.tensor(pred).cuda()

        gt = pix_gt.long()

        nounlabeled_pred = pred[gt != 255]
        nounlabeled_gt = gt[gt != 255]

        inter = histc(
            nounlabeled_pred[nounlabeled_gt == nounlabeled_pred].float().cpu(),
            bins=(num_classes),
            min=0,
            max=num_classes - 1)
        pred_area = histc(nounlabeled_pred.float().cpu(), bins=(num_classes), min=0, max=num_classes - 1)
        gt_area = histc(nounlabeled_gt.float().cpu(), bins=(num_classes), min=0, max=num_classes - 1)
        union = pred_area + gt_area - inter

        total_inter = total_inter + inter
        total_union = total_union + union

    total_inter = total_inter.cuda()
    total_union = total_union.cuda()

    all_reduce(total_inter)
    all_reduce(total_union)

    IoU = torch.nan_to_num(total_inter / (total_union + 1e-10), 0.0)
    return IoU.cpu().numpy()


def evaluate_wvisualize(model,
                        loader,
                        gt_label_filter=False,
                        crf=False,
                        scales=[1.0],
                        flip_directions=['none'],
                        output_key='pix_probs_cam',
                        show_folder='work_dirs/pascal_voc/show_ae_resnet50'):
    num_classes = loader.dataset.NUM_CLASSES
    total_inter = torch.zeros((num_classes, ))
    total_union = torch.zeros((num_classes, ))

    for n, dataset_dict in enumerate(tqdm(loader)):
        raw_img = dataset_dict['raw_img'].cuda()
        img = dataset_dict['img'].cuda()
        pix_gt = dataset_dict['pix_gt'].cuda()
        img_gt = dataset_dict['img_gt'].cuda()

        with torch.cuda.amp.autocast():
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

        nounlabeled_pred = pred[gt != 255]
        nounlabeled_gt = gt[gt != 255]

        inter = histc(
            nounlabeled_pred[nounlabeled_gt == nounlabeled_pred].float().cpu(),
            bins=(num_classes),
            min=0,
            max=num_classes - 1)
        pred_area = histc(nounlabeled_pred.float().cpu(), bins=(num_classes), min=0, max=num_classes - 1)
        gt_area = histc(nounlabeled_gt.float().cpu(), bins=(num_classes), min=0, max=num_classes - 1)
        union = pred_area + gt_area - inter

        total_inter = total_inter + inter
        total_union = total_union + union

        class_ids = torch.nonzero(img_gt[0])[:, 0].cpu().numpy().tolist()
        classes = [loader.dataset.CLASSES[idx + 1] for idx in class_ids]

        if show_folder is not None:
            if not osp.exists(show_folder):
                os.makedirs(show_folder, 0o775)
            filename = osp.splitext(dataset_dict['filename'][0])[0]
            raw_img = raw_img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            gt = gt.cpu().numpy().astype(np.uint8)
            pred = pred.cpu().numpy().astype(np.uint8)
            show_img = Image.fromarray(raw_img)
            show_gt = Image.fromarray(gt).convert('L')
            show_gt.putpalette(loader.dataset.get_palette())
            show_pred = Image.fromarray(pred).convert('L')
            show_pred.putpalette(loader.dataset.get_palette())
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
            plt.subplot(231)
            plt.imshow(np.array(show_img))
            plt.axis('off')
            plt.title('raw_img')
            plt.subplot(232)
            plt.imshow(np.array(show_gt.convert('RGB')))
            plt.axis('off')
            plt.title('gt')
            plt.subplot(233)
            plt.imshow(np.array(show_pred.convert('RGB')))
            plt.axis('off')
            iou = inter / (union + 1e-6)
            bg_iou = iou[0]
            fg_iou = torch.mean(iou[1:][img_gt[0].cpu() == 1])
            plt.title(f'bg_iou {int(bg_iou.cpu().numpy()*100)} fg_iou {int(fg_iou.cpu().numpy()*100)}')
            segments = slic(raw_img, n_segments=60, compactness=10)
            out = mark_boundaries(raw_img, segments)
            plt.subplot(223)
            plt.title("n_segments=60")
            plt.imshow(out)
            segments2 = slic(raw_img, n_segments=300, compactness=10)
            out2 = mark_boundaries(raw_img, segments2)
            plt.subplot(224)
            plt.title("n_segments=300")
            plt.imshow(out2)
            plt.suptitle(' '.join(classes))
            plt.tight_layout()
            plt.savefig(osp.join(show_folder, filename + '_show.png'))
            plt.close()

    total_inter = total_inter.cuda()
    total_union = total_union.cuda()

    all_reduce(total_inter)
    all_reduce(total_union)

    IoU = total_inter / total_union
    return IoU.cpu().numpy()
