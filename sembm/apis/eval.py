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
from ..utils.distributed import all_reduce, collect_results_cpu


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

        for k in dataset_dict.keys():
            if isinstance(dataset_dict[k], torch.Tensor):
                dataset_dict[k] = dataset_dict[k].cuda()

        pix_gt = dataset_dict['pix_gt']
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
                raw_img.cpu().numpy().astype(np.uint8),
                torch.softmax(pix_pred, dim=0).cpu().numpy(),
                t=10,
                scale_factor=1,
                labels=num_classes)
            pix_pred_crf = torch.tensor(pix_pred_crf, device=pix_pred.device)
            pix_pred = pix_pred_crf

        pred = torch.argmax(pix_pred, dim=0)
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


def inference(model, loader, crf=False, scales=[1.0], flip_directions=['none'], infer_folder=None):
    num_classes = loader.dataset.NUM_CLASSES

    assert infer_folder is not None

    if infer_folder is not None:
        if not osp.exists(infer_folder):
            os.makedirs(infer_folder, 0o775)

    for n, dataset_dict in enumerate(tqdm(loader)):

        for k in dataset_dict.keys():
            if isinstance(dataset_dict[k], torch.Tensor):
                dataset_dict[k] = dataset_dict[k].cuda()

        dataset_dict['scales'] = scales
        dataset_dict['flip_directions'] = flip_directions

        pix_pred = model(dataset_dict)
        pix_pred = pix_pred[0]

        if crf:
            raw_img = dataset_dict['raw_img'].cuda()
            raw_img = raw_img[0].permute(1, 2, 0).contiguous()
            pix_pred_crf = crf_inference(
                raw_img.cpu().numpy().astype(np.uint8),
                torch.softmax(pix_pred, dim=0).cpu().numpy(),
                t=10,
                scale_factor=1,
                labels=num_classes)
            pix_pred_crf = torch.tensor(pix_pred_crf, device=pix_pred.device)
            pix_pred = pix_pred_crf

        pred = torch.argmax(pix_pred, dim=0)

        pred = pred.cpu().numpy().astype(np.uint8)
        pred = Image.fromarray(pred).convert('L')
        pred.putpalette(loader.dataset.get_palette())

        filename = osp.splitext(dataset_dict['filename'][0])[0]

        pred.save(osp.join(infer_folder, filename + '.png'))

    return


def can_loss(pix_logit, img_gt):
    import torch.nn as nn
    class_ids = torch.nonzero(img_gt)[:, 0] + 1
    reverse_class_ids = torch.nonzero((1 - img_gt))[:, 0] + 1

    pix_logit = pix_logit.permute(1, 2, 0).contiguous()
    in_logit = pix_logit[:, :, class_ids]
    in_logit = torch.cat([pix_logit[:, :, :1], in_logit], dim=-1)
    out_logit = pix_logit[:, :, reverse_class_ids]

    _, rank_ids = torch.sort(in_logit, dim=-1, descending=True)

    topk_mask = torch.zeros_like(in_logit)
    ones_mask = torch.ones_like(in_logit)
    for i in range(len(class_ids) + 1):
        mask = rank_ids[:, :, i] == 0
        sel_in_pixs = in_logit[mask]

        if i == 0:
            sel_in_pixs, topk_ids = torch.topk(sel_in_pixs, k=1, dim=-1)
        else:
            sel_in_pixs, topk_ids = torch.topk(sel_in_pixs, k=i, dim=-1)

        topk_mask[mask] = torch.scatter(topk_mask[mask], dim=-1, index=topk_ids, src=ones_mask[mask])
        # print(i, topk_ids[2], topk_mask[mask][2], sel_in_pixs.shape)

    # in_logit, _ = torch.topk(in_logit, k=1, dim=0)
    in_logit = in_logit * topk_mask
    out_logit, _ = torch.topk(out_logit, k=1, dim=-1)
    local_loss = nn.Softplus()(torch.logsumexp(-in_logit, dim=-1) + torch.logsumexp(out_logit + 6, dim=-1))

    return local_loss


def can_loss_v2(pix_logits, img_gt):
    import torch.nn as nn
    m_corr = torch.randn((21, 21))
    tau = 0.2
    # Adaptive Group Split

    # Calculate IC group
    ic_class_ids = torch.nonzero(img_gt)[:, 0] + 1
    pix_probs = torch.softmax(pix_logits, dim=0)
    ic_gp_probs = pix_probs[ic_class_ids]
    ic_filter_mask = torch.zeros_like(pix_probs)
    ## Anchor Predictions and Indices (H x W)
    anchor_probs, anchor_ids = torch.max(ic_gp_probs, dim=0, keepdims=True)
    ## set anchor id as 1
    ic_filter_mask = torch.scatter(ic_filter_mask, dim=0, index=anchor_ids[None], src=torch.ones_like(ic_filter_mask))
    ## Filter useless classes in IC classes group

    sample_m_corr = torch.scatter(torch.zeros_like(pix_probs)[:, None], dim=1, index=anchor_ids, src=m_corr)[:, 0]
    cond = (anchor_probs[None] - sample_m_corr * pix_probs) < tau
    ic_filter_mask[cond] = 1
    ic_gp_logits = (pix_logits * ic_filter_mask)[ic_class_ids]

    # Calculate OC group
    oc_class_ids = torch.cat([0, torch.nonzero((1 - img_gt))[:, 0] + 1])
    oc_gp_logits = pix_logits[oc_class_ids]

    local_loss = nn.Softplus()(torch.logsumexp(-ic_gp_logits, dim=-1) + torch.logsumexp(oc_gp_logits + 6, dim=-1))

    return local_loss


import cv2


def min_max_norm(feat):
    min_v = np.min(feat)
    max_v = np.max(feat)
    return (feat - min_v) / (max_v - min_v)


def color_map(mask):
    # tmp = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # cv2.COLORMAP_VIRIDIS
    tmp = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    return tmp


def IoU_cal(pred, gt):
    inter = pred[pred == gt]
    area_inter = torch.histc(inter.float(), bins=2, min=0, max=1)
    area_preds = torch.histc(pred.float(), bins=2, min=0, max=1)
    area_targets = torch.histc(gt.float(), bins=2, min=0, max=1)
    area_union = area_preds + area_targets - area_inter

    area_inter = area_inter[1]
    area_union = area_union[1]

    if area_union == 0 or area_inter == 0:
        return 0.0
    else:
        return float((area_inter / area_union).cpu().numpy())


def applyCustomColorMap(im_gray):
    import seaborn as sns
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    cmap = np.array(sns.color_palette('rocket'))
    cmap = (cmap * 255).astype(np.uint8)
    cmap = torch.tensor(cmap).permute(1, 0)[None].float()
    cmap = F.interpolate(cmap, 256, mode='linear')[0].cpu().numpy().astype(np.uint8)

    # Red
    # lut[:, 0, 0] = [
    #     255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    #     255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    #     255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 253,
    #     251, 249, 247, 245, 242, 241, 238, 237, 235, 233, 231, 229, 227, 225, 223, 221, 219, 217, 215, 213, 211, 209,
    #     207, 205, 203, 201, 199, 197, 195, 193, 191, 189, 187, 185, 183, 181, 179, 177, 175, 173, 171, 169, 167, 165,
    #     163, 161, 159, 157, 155, 153, 151, 149, 147, 145, 143, 141, 138, 136, 134, 132, 131, 129, 126, 125, 122, 121,
    #     118, 116, 115, 113, 111, 109, 107, 105, 102, 100, 98, 97, 94, 93, 91, 89, 87, 84, 83, 81, 79, 77, 75, 73, 70,
    #     68, 66, 64, 63, 61, 59, 57, 54, 52, 51, 49, 47, 44, 42, 40, 39, 37, 34, 33, 31, 29, 27, 25, 22, 20, 18, 17, 14,
    #     13, 11, 9, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # ]
    lut[:, 0, 0] = cmap[0]

    # Green
    # lut[:, 0, 1] = [
    #     255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    #     255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    #     255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    #     255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    #     255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    #     255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 252, 250, 248,
    #     246, 244, 242, 240, 238, 236, 234, 232, 230, 228, 226, 224, 222, 220, 218, 216, 214, 212, 210, 208, 206, 204,
    #     202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178, 176, 174, 171, 169, 167, 165, 163, 161, 159,
    #     157, 155, 153, 151, 149, 147, 145, 143, 141, 139, 137, 135, 133, 131, 129, 127, 125, 123, 121, 119, 117, 115,
    #     113, 111, 109, 107, 105, 103, 101, 99, 97, 95, 93, 91, 89, 87, 85, 83, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64,
    #     62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8,
    #     6, 4, 2, 0
    # ]
    lut[:, 0, 1] = cmap[1]

    # Blue
    # lut[:, 0, 2] = [
    #     195, 194, 193, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 179, 178, 177, 176, 175, 174, 173, 172,
    #     171, 170, 169, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 155, 154, 153, 152, 151, 150, 149, 148,
    #     147, 146, 145, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 131, 130, 129, 128, 127, 126, 125, 125,
    #     125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
    #     126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
    #     126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    #     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    #     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    #     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    #     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 126, 126, 126, 126, 126, 126, 126, 126, 126,
    #     126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
    #     126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126
    # ]
    lut[:, 0, 2] = cmap[2]

    # Apply custom colormap through LUT
    im_color = cv2.LUT(cv2.cvtColor(im_gray, cv2.COLOR_GRAY2RGB), lut)

    return im_color


def evaluate_wvisualize(model,
                        loader,
                        gt_label_filter=False,
                        crf=False,
                        scales=[1.0],
                        flip_directions=['none'],
                        show_folder='work_dirs/voc_seam_gt/deeplabv1_seam_can_sgd-lr7e-4_e30_bs16_top1leqtop1/show'):
    num_classes = loader.dataset.NUM_CLASSES
    total_inter = torch.zeros((num_classes, ))
    total_union = torch.zeros((num_classes, ))

    for n, dataset_dict in enumerate(tqdm(loader)):

        for k in dataset_dict.keys():
            if isinstance(dataset_dict[k], torch.Tensor):
                dataset_dict[k] = dataset_dict[k].cuda()

        # VOC
        # name_list = [
        #     "2007_001586", "2007_001955", "2007_002643", "2007_002719", "2007_002852", "2007_003110", "2007_003137",
        #     "2007_004190", "2007_00897", "2011_002178", "2011_001713", "2011_001529", "2011_001071", "2011_000051",
        #     "2010_005252", "2010_005180", "2010_001351", "2010_000724", "2009_005231", "2009_000771", "2009_000012",
        #     "2008_005105"
        # ]

        # if osp.splitext(dataset_dict['filename'][0])[0] not in name_list:
        #     continue

        # VOC PSA
        # if osp.splitext(dataset_dict['filename'][0])[0] not in ['2010_000256', '2011_001071']:
        #     continue

        # VOC SEAM
        # if osp.splitext(dataset_dict['filename'][0])[0] not in ['2009_000771', '2009_003849', '2010_005252']:
        #     continue

        # VOC MCTformer
        # if osp.splitext(dataset_dict['filename'][0])[0] not in ['2008_004621', '2010_001264']:
        #     continue

        # COCO PSA
        # if osp.splitext(dataset_dict['filename'][0])[0] not in ['COCO_val2014_000000143901']:
        #     #         'COCO_val2014_000000000073', 'COCO_val2014_000000060202', 'COCO_val2014_000000192607',
        #     #         'COCO_val2014_000000207635', 'COCO_val2014_000000431136', 'COCO_val2014_000000530220',
        #     #         'COCO_val2014_000000561357', 'COCO_val2014_000000143901', 'COCO_val2014_000000225014',
        #     #         'COCO_val2014_000000302155'
        #     # ]:
        #     continue

        # COCO SEAM
        # if osp.splitext(dataset_dict['filename'][0])[0] not in ['COCO_val2014_000000050117', 'COCO_val2014_000000099581']:
        #     continue

        # COCO MCTformer
        # COCO_val2014_000000126123_eval
        # if osp.splitext(dataset_dict['filename'][0])[0] not in ['COCO_val2014_000000126123', 'COCO_val2014_000000205834', 'COCO_val2014_000000225731']:
        #     continue

        # loss map
        # if osp.splitext(dataset_dict['filename'][0])[0] not in [
        #         '2007_009897', '2008_007273', '2010_001962', '2011_003197', '2008_006063', '2008_001074', '2007_007810'
        # ]:
        #     continue

        # VOC PSA normal cases
        # if osp.splitext(dataset_dict['filename'][0])[0] not in ['2007_002094', '2010_005118', '2007_002565']:
        #     continue

        # VOC SEAM normal cases
        # if osp.splitext(dataset_dict['filename'][0])[0] not in ['2007_000676', '2008_000700', '2008_002536', '2007_008260']:
        #     continue

        # VOC MCTformer normal cases
        # if osp.splitext(dataset_dict['filename'][0])[0] not in [
        #         '2007_001568', '2007_002046', '2007_006866', '2007_002823', '2007_005354', '2007_006364', '2007_005845'
        # ]:
        #     continue

        # COCO PSA normal cases
        # if osp.splitext(dataset_dict['filename'][0])[0] not in ['COCO_val2014_000000060687', 'COCO_val2014_000000066800', 'COCO_val2014_000000074711']:
        #     continue

        # COCO SEAM normal cases
        # if osp.splitext(dataset_dict['filename'][0])[0] not in [
        #         'COCO_val2014_000000003926', 'COCO_val2014_000000520727', 'COCO_val2014_000000554735',
        #         'COCO_val2014_000000578703'
        # ]:
        #     continue

        # COCO MCTformer normal cases
        # if osp.splitext(dataset_dict['filename'][0])[0] not in [
        #         'COCO_val2014_000000009426', 'COCO_val2014_000000039072', 'COCO_val2014_000000489745', 'COCO_val2014_000000241876'
        # ]:
        #     continue

        img_gt = dataset_dict['img_gt'].cuda()
        pix_gt = dataset_dict['pix_gt'].cuda()
        pseudo_pix_gt = dataset_dict['pseudo_pix_gt'].cuda()
        raw_img = dataset_dict['raw_img'].cuda()
        dataset_dict['gt_label_filter'] = gt_label_filter
        dataset_dict['scales'] = scales
        dataset_dict['flip_directions'] = flip_directions

        pix_pred = model(dataset_dict)

        pix_pred = pix_pred[0]
        pseudo_pix_gt = pseudo_pix_gt[0]
        pix_gt = pix_gt[0]
        img_gt = img_gt[0]
        raw_img = raw_img[0]

        if crf:
            _raw_img = raw_img.permute(1, 2, 0).contiguous()
            pix_pred_crf = crf_inference(
                _raw_img.cpu().numpy().astype(np.uint8),
                torch.softmax(pix_pred, dim=0).cpu().numpy(),
                t=10,
                scale_factor=1,
                labels=21)
            pix_pred_crf = torch.tensor(pix_pred_crf, device=pix_pred.device)
            pix_pred = pix_pred_crf

        loss_map = can_loss(pix_pred, img_gt)

        clean_pix_pred = pix_pred.clone()
        clean_pix_pred[1:] = clean_pix_pred[1:] * img_gt[:, None, None]

        pred = torch.argmax(pix_pred, dim=0)
        clean_pred = torch.argmax(clean_pix_pred, dim=0)
        gt = pix_gt.long()
        pseudo_gt = pseudo_pix_gt.long()

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

        # class_ids = torch.nonzero(img_gt)[:, 0] + 1
        # classes = [loader.dataset.CLASSES[idx] for idx in class_ids]

        class_ids = torch.unique(pix_gt).long()
        class_ids = class_ids[class_ids != 0]
        class_ids = class_ids[class_ids != 255]

        pseudo_class_ids = torch.unique(pseudo_gt)

        img_gt = torch.zeros_like(img_gt)
        img_gt[class_ids - 1] = 1

        pred_class_ids = torch.unique(pred)
        pred_class_ids = pred_class_ids[pred_class_ids != 0]
        pred_class_mask = torch.zeros_like(img_gt)
        pred_class_mask[pred_class_ids - 1] = 1
        norm_class_ids = torch.nonzero(img_gt * pred_class_mask)[:, 0] + 1
        abnormal_class_ids = torch.nonzero((1 - img_gt) * pred_class_mask)[:, 0] + 1
        pred_classes = [loader.dataset.CLASSES[idx] for idx in pred_class_ids]

        # print(pred_class_ids, class_ids)
        # mask = torch.zeros_like(pred)
        # mask[pred == 8] = 255
        # alpha_mask = mask.cpu().numpy()[:, :, None].astype(np.uint8)
        # _raw_img = raw_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # _img = np.concatenate([_raw_img, alpha_mask], axis=-1)
        # print(_img.shape, _img.dtype)
        # Image.fromarray(_img).save('try.png')
        # exit(0)

        total_iou = (inter / union)[class_ids]
        # tiou = float(torch.min(total_iou).cpu().numpy())
        iou = IoU_cal(nounlabeled_pred > 0, nounlabeled_gt > 0)
        # if total_iou > 0.8 or iou < 0.6:
        #     continue
        # cond = len(class_ids) >= 1 and len(abnormal_class_ids) >= 1 and torch.sum(
        #     total_iou < 0.6) == 0 and iou > 0.7
        # cond2 = torch.sum(pred == abnormal_class_ids[0]) > 1500
        # if not (cond and cond2):
        #     continue
        cond = len(class_ids) > 1 and len(abnormal_class_ids) == 0 and iou > 0.6 and torch.sum(total_iou < 0.6) == 0
        if not cond:
            continue

        # if torch.sum(abnormal_class_ids) != 0 and len(norm_class_ids) == len(class_ids):
        # filename = osp.splitext(dataset_dict['filename'][0])[0]
        # loss_map_np = loss_map.cpu().numpy()
        # loss_map_np_color = color_map(min_max_norm(loss_map_np))
        # raw_img_np = raw_img.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
        # pred_np = pred.cpu().numpy().astype(np.uint8)
        # gt_np = gt.cpu().numpy().astype(np.uint8)
        # pred_pil = Image.fromarray(pred_np)
        # gt_pil = Image.fromarray(gt_np)
        # pred_pil.putpalette(loader.dataset.get_palette())
        # gt_pil.putpalette(loader.dataset.get_palette())
        # pred_pil = np.array(pred_pil.convert('RGB'))
        # gt_pil = np.array(gt_pil.convert('RGB'))
        # import matplotlib.pyplot as plt
        # plt.subplot(221)
        # plt.imshow(raw_img_np)
        # plt.subplot(222)
        # _loss_map_np = (min_max_norm(loss_map_np) * 255).astype(np.uint8)
        # _loss_map_np[(clean_pred == pred).cpu().numpy()] = 0
        # loss_map_np_color = applyCustomColorMap(_loss_map_np)
        # canvas = cv2.addWeighted(loss_map_np_color, 0.7, raw_img_np, 0.3, 1.0)
        # Image.fromarray(canvas).save(f'{show_folder}/{filename}_cut_loss_map.png')
        # _loss_map_np = (min_max_norm(loss_map_np) * 255).astype(np.uint8)
        # loss_map_np_color = applyCustomColorMap(_loss_map_np)
        # canvas = cv2.addWeighted(loss_map_np_color, 0.7, raw_img_np, 0.3, 1.0)
        # Image.fromarray(canvas).save(f'{show_folder}/{filename}_loss_map.png')
        # plt.imshow(canvas)
        # plt.subplot(223)
        # plt.imshow(pred_pil)
        # plt.subplot(224)
        # plt.imshow(gt_pil)
        # plt.savefig(f'{show_folder}/{filename}.png')
        # plt.close()
        # exit(0)
        # if True:
        # total_iou = (inter / union)[class_ids]
        # tiou = float(torch.min(total_iou).cpu().numpy())
        # iou = IoU_cal(nounlabeled_pred > 0, nounlabeled_gt > 0)
        if show_folder is not None:
            if not osp.exists(show_folder):
                os.makedirs(show_folder, 0o775)
        filename = osp.splitext(dataset_dict['filename'][0])[0]
        raw_img = raw_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        gt = gt.cpu().numpy().astype(np.uint8)
        pseudo_gt = pseudo_gt.cpu().numpy().astype(np.uint8)
        pred = pred.cpu().numpy().astype(np.uint8)
        clean_pred = clean_pred.cpu().numpy().astype(np.uint8)
        show_img = Image.fromarray(raw_img)
        show_gt = Image.fromarray(gt).convert('L')
        show_gt.putpalette(loader.dataset.get_palette())
        show_pseudo_gt = Image.fromarray(pseudo_gt).convert('L')
        show_pseudo_gt.putpalette(loader.dataset.get_palette())
        show_clean_pred = Image.fromarray(clean_pred).convert('L')
        show_clean_pred.putpalette(loader.dataset.get_palette())
        show_pred = Image.fromarray(pred).convert('L')
        show_pred.putpalette(loader.dataset.get_palette())
        show_img.save(osp.join(show_folder, filename + '_img.png'))
        show_gt.save(osp.join(show_folder, filename + '_gt.png'))
        show_pseudo_gt.save(osp.join(show_folder, filename + '_pgt.png'))
        show_clean_pred.save(osp.join(show_folder, filename + '_clean_pred.png'))
        show_pred.save(osp.join(show_folder, filename + '_pred.png'))
        ROW = 1
        COL = 5
        UNIT_WIDTH_SIZE = show_img.width
        UNIT_HEIGHT_SIZE = show_img.height
        target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL + 5 * (COL - 1), UNIT_HEIGHT_SIZE * ROW), (255, 255, 255))
        target.paste(show_img, (UNIT_WIDTH_SIZE * 0, UNIT_HEIGHT_SIZE * 0))
        target.paste(show_gt, (UNIT_WIDTH_SIZE * 1 + 5, UNIT_HEIGHT_SIZE * 0))
        target.paste(show_pseudo_gt, (UNIT_WIDTH_SIZE * 2 + 10, UNIT_HEIGHT_SIZE * 0))
        target.paste(show_pred, (UNIT_WIDTH_SIZE * 3 + 15, UNIT_HEIGHT_SIZE * 0))
        target.paste(show_clean_pred, (UNIT_WIDTH_SIZE * 4 + 20, UNIT_HEIGHT_SIZE * 0))
        target.save(osp.join(show_folder, filename + '_eval.png'))
        # target.save(osp.join(show_folder, filename + f'_{iou*100:.2f}_{tiou*100:.2f}_eval.png'))

    total_inter = total_inter.cuda()
    total_union = total_union.cuda()

    all_reduce(total_inter)
    all_reduce(total_union)

    IoU = total_inter / total_union
    return IoU.cpu().numpy()


# def evaluate_wvisualize(model,
#                         loader,
#                         gt_label_filter=False,
#                         crf=False,
#                         scales=[1.0],
#                         flip_directions=['none'],
#                         show_folder='work_dirs/voc_seam_gt/deeplabv1_seam_can_sgd-lr7e-4_e30_bs16_top1leqtop1/show'):
#     num_classes = loader.dataset.NUM_CLASSES
#     total_inter = torch.zeros((num_classes, ))
#     total_union = torch.zeros((num_classes, ))

#     # filenames = [osp.splitext(x)[0] for x in os.listdir('work_dirs/psa_fig1')]

#     for n, dataset_dict in enumerate(tqdm(loader)):

#         for k in dataset_dict.keys():
#             if isinstance(dataset_dict[k], torch.Tensor):
#                 dataset_dict[k] = dataset_dict[k].cuda()

#         img = dataset_dict['img'].cuda()
#         img_gt = dataset_dict['img_gt'].cuda()
#         pix_gt = dataset_dict['pix_gt'].cuda()
#         pseudo_pix_gt = dataset_dict['pseudo_pix_gt'].cuda()
#         raw_img = dataset_dict['raw_img'].cuda()
#         name = dataset_dict['filename']
#         dataset_dict['gt_label_filter'] = gt_label_filter
#         dataset_dict['scales'] = scales
#         dataset_dict['flip_directions'] = flip_directions

#         # 2007_001586, 2007_004112
#         # if osp.splitext(name[0])[0] not in ['2007_001585', '2009_001536', '2008_004612', '2009_000418']:
#         #     continue
#         # if osp.splitext(dataset_dict['filename'][0])[0] not in ['2007_000452']:
#         #     continue

#         # if osp.splitext(dataset_dict['filename'][0])[0] not in ['2007_000572', '2007_000559', '2007_000762', '2007_002400']:
#         #     continue

#         # if osp.splitext(dataset_dict['filename'][0])[0] not in ['2009_000012']:
#         #     continue

#         # if osp.splitext(name[0])[0] not in filenames:
#         #     continue

#         # if osp.splitext(dataset_dict['filename'][0])[0] not in ['2007_003714']:
#         #     continue

#         pix_pred = model(dataset_dict)

#         pix_pred = pix_pred[0]
#         pseudo_pix_gt = pseudo_pix_gt[0]
#         pix_gt = pix_gt[0]
#         img_gt = img_gt[0]
#         raw_img = raw_img[0]

#         if crf:
#             _raw_img = raw_img.permute(1, 2, 0).contiguous()
#             pix_pred_crf = crf_inference(
#                 _raw_img.cpu().numpy().astype(np.uint8),
#                 torch.softmax(pix_pred, dim=0).cpu().numpy(),
#                 t=10,
#                 scale_factor=1,
#                 labels=21)
#             pix_pred_crf = torch.tensor(pix_pred_crf, device=pix_pred.device)
#             pix_pred = pix_pred_crf

#         local_loss = can_loss(pix_pred, img_gt)

#         clean_pix_pred = pix_pred.clone()
#         clean_pix_pred[1:] = clean_pix_pred[1:] * img_gt[:, None, None]

#         pred = torch.argmax(pix_pred, dim=0)
#         clean_pred = torch.argmax(clean_pix_pred, dim=0)
#         gt = pix_gt.long()
#         pseudo_gt = pseudo_pix_gt.long()

#         # local_loss[clean_pred == pred] *= 0.3
#         local_loss = local_loss.cpu().numpy()
#         local_loss = min_max_norm(local_loss)
#         # local_loss[(clean_pred == pred).cpu().numpy()] *= 0
#         local_loss_map = color_map(local_loss)
#         # local_loss_map[(clean_pred == pred).cpu().numpy()] = (0, 0, 0)

#         print(pix_pred[:, clean_pred != pred])

#         raw_img = raw_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

#         raw_img = local_loss_map * 0.7 + raw_img * 0.3
#         raw_img = raw_img.astype(np.uint8)
#         # raw_img = cv2.addWeighted(local_loss_map, 0.5, raw_img, 0.5, 1.0)

#         Image.fromarray(raw_img).save('3.png')

#         exit(0)

#         nounlabeled_pred = pred[gt != 255]
#         nounlabeled_gt = gt[gt != 255]

#         inter = histc(
#             nounlabeled_pred[nounlabeled_gt == nounlabeled_pred].float().cpu(),
#             bins=(num_classes),
#             min=0,
#             max=num_classes - 1)
#         pred_area = histc(nounlabeled_pred.float().cpu(), bins=(num_classes), min=0, max=num_classes - 1)
#         gt_area = histc(nounlabeled_gt.float().cpu(), bins=(num_classes), min=0, max=num_classes - 1)
#         union = pred_area + gt_area - inter

#         total_inter = total_inter + inter
#         total_union = total_union + union

#         class_ids = torch.nonzero(img_gt)[:, 0] + 1
#         classes = [loader.dataset.CLASSES[idx] for idx in class_ids]

#         pseudo_class_ids = torch.unique(pseudo_gt)

#         pred_class_ids = torch.unique(pred)
#         pred_class_ids = pred_class_ids[pred_class_ids != 0]
#         pred_class_mask = torch.zeros_like(img_gt)
#         pred_class_mask[pred_class_ids - 1] = 1
#         norm_class_ids = torch.nonzero(img_gt * pred_class_mask)[:, 0] + 1
#         abnormal_class_ids = torch.nonzero((1 - img_gt) * pred_class_mask)[:, 0] + 1
#         pred_classes = [loader.dataset.CLASSES[idx] for idx in pred_class_ids]

#         # print(pred_class_ids, class_ids)
#         # mask = torch.zeros_like(pred)
#         # mask[pred == 8] = 255
#         # alpha_mask = mask.cpu().numpy()[:, :, None].astype(np.uint8)
#         # _raw_img = raw_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
#         # _img = np.concatenate([_raw_img, alpha_mask], axis=-1)
#         # print(_img.shape, _img.dtype)
#         # Image.fromarray(_img).save('try.png')
#         # exit(0)

#         # print(classes, pred_classes)
#         # print(class_ids, pred_class_ids, norm_class_ids, abnormal_class_ids)
#         # if True:
#         # if torch.sum(abnormal_class_ids) != 0 and len(norm_class_ids) == len(class_ids):
#         # if len(pseudo_class_ids) <= len(class_ids):
#         if True:
#             # if torch.sum(abnormal_class_ids) != 0:
#             # if 'chair' in classes and len(classes) == 1:
#             # print(classes, pred_classes)
#             if show_folder is not None:
#                 if not osp.exists(show_folder):
#                     os.makedirs(show_folder, 0o775)
#                 filename = osp.splitext(dataset_dict['filename'][0])[0]
#                 raw_img = raw_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
#                 gt = gt.cpu().numpy().astype(np.uint8)
#                 pseudo_gt = pseudo_gt.cpu().numpy().astype(np.uint8)
#                 print(pseudo_gt)
#                 # print(pseudo_gt.shape)
#                 pred = pred.cpu().numpy().astype(np.uint8)
#                 clean_pred = clean_pred.cpu().numpy().astype(np.uint8)
#                 show_img = Image.fromarray(raw_img)
#                 show_gt = Image.fromarray(gt).convert('L')
#                 show_gt.putpalette(loader.dataset.get_palette())
#                 show_pseudo_gt = Image.fromarray(pseudo_gt).convert('L')
#                 show_pseudo_gt.putpalette(loader.dataset.get_palette())
#                 show_clean_pred = Image.fromarray(clean_pred).convert('L')
#                 show_clean_pred.putpalette(loader.dataset.get_palette())
#                 show_pred = Image.fromarray(pred).convert('L')
#                 show_pred.putpalette(loader.dataset.get_palette())
#                 show_img.save(osp.join(show_folder, filename + '_img.png'))
#                 show_gt.save(osp.join(show_folder, filename + '_gt.png'))
#                 show_pseudo_gt.save(osp.join(show_folder, filename + '_pgt.png'))
#                 show_clean_pred.save(osp.join(show_folder, filename + '_clean_pred.png'))
#                 show_pred.save(osp.join(show_folder, filename + '_pred.png'))
#                 ROW = 1
#                 COL = 5
#                 UNIT_WIDTH_SIZE = show_img.width
#                 UNIT_HEIGHT_SIZE = show_img.height
#                 target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL + 5 * (COL - 1), UNIT_HEIGHT_SIZE * ROW),
#                                    (255, 255, 255))
#                 target.paste(show_img, (UNIT_WIDTH_SIZE * 0, UNIT_HEIGHT_SIZE * 0))
#                 target.paste(show_gt, (UNIT_WIDTH_SIZE * 1 + 5, UNIT_HEIGHT_SIZE * 0))
#                 target.paste(show_pseudo_gt, (UNIT_WIDTH_SIZE * 2 + 10, UNIT_HEIGHT_SIZE * 0))
#                 target.paste(show_pred, (UNIT_WIDTH_SIZE * 3 + 15, UNIT_HEIGHT_SIZE * 0))
#                 target.paste(show_clean_pred, (UNIT_WIDTH_SIZE * 4 + 20, UNIT_HEIGHT_SIZE * 0))
#                 target.save(osp.join(show_folder, filename + '.png'))

#     total_inter = total_inter.cuda()
#     total_union = total_union.cuda()

#     all_reduce(total_inter)
#     all_reduce(total_union)

#     IoU = total_inter / total_union
#     return IoU.cpu().numpy()


def _evaluate_wvisualize(model,
                         loader,
                         gt_label_filter=False,
                         crf=False,
                         scales=[1.0],
                         flip_directions=['none'],
                         show_folder='work_dirs/voc_seam_gt/deeplabv1_seam_can_sgd-lr7e-4_e30_bs16_top1leqtop1/show'):
    num_classes = loader.dataset.NUM_CLASSES
    total_inter = torch.zeros((num_classes, ))
    total_union = torch.zeros((num_classes, ))

    for n, dataset_dict in enumerate(tqdm(loader)):

        img_gt = dataset_dict['img_gt'].cuda()
        pix_gt = dataset_dict['pix_gt'].cuda()
        raw_img = dataset_dict['raw_img'].cuda()
        dataset_dict['gt_label_filter'] = gt_label_filter
        dataset_dict['scales'] = scales
        dataset_dict['flip_directions'] = flip_directions

        pix_pred = model(dataset_dict)
        pix_pred = pix_pred[0]
        pix_gt = pix_gt[0]

        if crf:
            _raw_img = raw_img[0].permute(1, 2, 0).contiguous()
            pix_pred_crf = crf_inference(
                _raw_img.cpu().numpy().astype(np.uint8),
                torch.softmax(pix_pred, dim=0).cpu().numpy(),
                t=10,
                scale_factor=1,
                labels=21)
            pix_pred_crf = torch.tensor(pix_pred_crf, device=pix_pred.device)
            pix_pred = pix_pred_crf

        pred = torch.argmax(pix_pred, dim=0)

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


def evaluate_debug(model, loader, gt_label_filter=False, crf=False, scales=[1.0], flip_directions=['none']):
    num_classes = loader.dataset.NUM_CLASSES
    total_inter = torch.zeros((num_classes, ))
    total_union = torch.zeros((num_classes, ))

    noisy_logits = []
    clean_logits = []

    for n, dataset_dict in enumerate(tqdm(loader)):

        raw_img = dataset_dict['raw_img'].cuda()
        img_gt = dataset_dict['img_gt'].cuda()
        pix_gt = dataset_dict['pix_gt'].cuda()
        pseudo_pix_gt = dataset_dict['pseudo_pix_gt'].cuda()
        name = dataset_dict['filename']
        dataset_dict['gt_label_filter'] = gt_label_filter
        dataset_dict['scales'] = scales
        dataset_dict['flip_directions'] = flip_directions

        if name[0] in ['2007_000346.jpg', '2007_000452.jpg']:
            continue

        # if name[0] not in []:
        #     continue

        pix_pred = model(dataset_dict)
        pix_pred = pix_pred[0]
        pix_gt = pix_gt[0]
        pseudo_pix_gt = pseudo_pix_gt[0]
        img_gt = img_gt[0]

        if crf:
            raw_img = dataset_dict['raw_img'].cuda()
            raw_img = raw_img[0].permute(1, 2, 0).contiguous()
            pix_pred_crf = crf_inference(
                raw_img.cpu().numpy().astype(np.uint8), pix_pred.cpu().numpy(), t=10, scale_factor=1, labels=21)
            pix_pred_crf = torch.tensor(pix_pred_crf, device=pix_pred.device)
            pix_pred = pix_pred_crf

        # class_ids = torch.nonzero(img_gt)[:, 0] + 1
        # reverse_class_ids = torch.nonzero((1 - img_gt))[:, 0] + 1

        # candidate_logits = pix_pred[class_ids, :, :]
        # nocandidate_logits = pix_pred[reverse_class_ids, :, :]

        # print(torch.mean(candidate_logits), torch.mean(nocandidate_logits))

        # import time

        # start = time.time()

        # ppt = pseudo_pix_gt.clone().long()
        # pt = pix_gt.clone().long()

        # ppt[ppt == 255] = 0
        # pt[pt == 255] = 0

        # noisy_logit = torch.gather(pix_pred, dim=0, index=ppt[None])[0]
        # clean_logit = torch.gather(pix_pred, dim=0, index=pt[None])[0]
        # noisy_cond = (pt != 0) * (pix_gt != 255)
        # clean_cond = (pt != 0) * (pix_gt != 255)
        # noisy_logits.append(noisy_logit[noisy_cond].cpu().numpy())
        # clean_logits.append(clean_logit[clean_cond].cpu().numpy())

        pred = torch.argmax(pix_pred, dim=0)

        # end = time.time()
        # print(f'{end - start}s')

        # name = dataset_dict['filename']
        # pred = np.array(
        #     Image.open(
        #         osp.join('../MCTformer/work_dirs/MCTformer_v2_official/pgt-psa-rw', name[0].replace('.jpg', '.png'))))
        # pred = torch.tensor(pred).cuda()

        gt = pix_gt.long()

        reverse_img_gt = 1 - img_gt
        class_ids = torch.nonzero(img_gt)[:, 0] + 1
        reverse_class_ids = torch.nonzero((1 - img_gt))[:, 0] + 1

        img_logits = pix_pred.mean((-2, -1))
        pred_onehot = F.one_hot(pred, num_classes=21)
        count = torch.sum(pred_onehot[:, :, 1:] * reverse_img_gt[None, None], dim=-1)

        FP_class_ids = torch.unique(pred)
        FP_class_ids = FP_class_ids[FP_class_ids != 0]

        torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=300, profile=None)

        if torch.sum(count) > 0:
            print(class_ids)
            print(FP_class_ids)
            print(reverse_class_ids)
            print(img_logits[class_ids])
            print(img_logits[FP_class_ids])
            print(img_logits[reverse_class_ids])
            fp_pixs = pix_pred.permute(1, 2, 0)[pred == 17]
            print(len(fp_pixs))
            print(torch.cat([fp_pixs[:, :1], fp_pixs[:, FP_class_ids]], dim=1))
            class_ids = torch.nonzero(img_gt)[:, 0].cpu().numpy().tolist()
            classes = [loader.dataset.CLASSES[idx + 1] for idx in class_ids]
            FP_class_ids = torch.unique(pred)
            FP_class_ids = FP_class_ids[FP_class_ids != 0] - 1
            FP_classes = [loader.dataset.CLASSES[idx + 1] for idx in FP_class_ids]
            import matplotlib.pyplot as plt
            plt.subplot(221)
            plt.imshow(raw_img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            plt.subplot(222)
            plt.imshow(pix_gt.cpu().numpy())
            plt.title(' '.join(classes))
            plt.subplot(223)
            plt.imshow(pred.cpu().numpy() == 17)
            plt.title(' '.join(FP_classes))
            plt.subplot(224)
            plt.imshow(pseudo_pix_gt.cpu().numpy())

            plt.savefig(f'{osp.splitext(name[0])[0]}.png')

            exit(0)

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

    # import pickle as pkl
    # pkl.dump(noisy_logits, open('neg_noisy_logits.pkl', 'wb'))
    # pkl.dump(clean_logits, open('neg_clean_logits.pkl', 'wb'))

    all_reduce(total_inter)
    all_reduce(total_union)

    IoU = torch.nan_to_num(total_inter / (total_union + 1e-10), 0.0)
    return IoU.cpu().numpy()


# def inference(model, loader, crf=False, scales=[1.0], flip_directions=['none']):
#     collect = []

#     loss_pixels = []

#     for n, dataset_dict in enumerate(tqdm(loader)):

#         items = []
#         raw_img = dataset_dict['raw_img'].cuda()
#         pix_gt = dataset_dict['pix_gt'].cuda()
#         pseudo_pix_gt = dataset_dict['pseudo_pix_gt'].cuda()
#         name = dataset_dict['filename']
#         dataset_dict['scales'] = scales
#         dataset_dict['flip_directions'] = flip_directions

#         pix_pred = model(dataset_dict)

#         redu_rloss = F.cross_entropy(pix_pred, pix_gt.long(), ignore_index=255)
#         rloss = F.cross_entropy(pix_pred, pix_gt.long(), ignore_index=255, reduction='none')

#         rloss[pix_gt == 255] = torch.max(rloss)

#         items.extend([redu_rloss.item(), torch.var(rloss).item(), torch.min(rloss).item(), torch.max(rloss).item()])

#         redu_ploss = F.cross_entropy(pix_pred, pseudo_pix_gt.long(), ignore_index=255)
#         ploss = F.cross_entropy(pix_pred, pseudo_pix_gt.long(), ignore_index=255, reduction='none')

#         items.extend([redu_ploss.item(), torch.var(ploss).item(), torch.min(ploss).item(), torch.max(ploss).item()])

#         pred = torch.argmax(pix_pred, dim=1)

#         import matplotlib.pyplot as plt
#         plt.subplot(231)
#         plt.imshow(raw_img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
#         plt.subplot(232)
#         plt.imshow(rloss[0].cpu().numpy())
#         plt.subplot(233)
#         plt.imshow(ploss[0].cpu().numpy())
#         plt.subplot(234)
#         plt.imshow(pred[0].cpu().numpy())
#         plt.subplot(235)
#         plt.imshow(pix_gt[0].cpu().numpy())
#         plt.subplot(236)
#         plt.imshow(pseudo_pix_gt[0].cpu().numpy())
#         plt.savefig(f'loss_show/{osp.splitext(name[0])[0]}.png')

#         riou = calculate_iou(pred[0] > 0, pix_gt[0] > 0, 2)
#         piou = calculate_iou(pred[0] > 0, pseudo_pix_gt[0] > 0, 2)

#         items.extend([riou[0].item(), riou[1].item(), piou[0].item(), piou[1].item()])

#         collect.append(items)

#     result = collect_results_cpu(collect, len(loader.dataset), None)

#     return result
