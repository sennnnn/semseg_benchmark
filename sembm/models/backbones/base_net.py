import numpy as np

import torch
import torch.nn as nn


class Normalize():

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def undo(self, imgarr):
        proc_img = imgarr.copy()

        proc_img[..., 0] = (self.std[0] * imgarr[..., 0] + self.mean[0]) * 255.
        proc_img[..., 1] = (self.std[1] * imgarr[..., 1] + self.mean[1]) * 255.
        proc_img[..., 2] = (self.std[2] * imgarr[..., 2] + self.mean[2]) * 255.

        return proc_img

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class BaseNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.normalize = Normalize()

        self.not_training = []  # freezing parameters
        self.bn_frozen = []  # freezing running stats
        self.from_scratch_layers = []  # new layers -> higher LR

    def _init_weights(self, path_to_weights):
        print("Loading weights from: ", path_to_weights)
        weights_dict = torch.load(path_to_weights)
        self.load_state_dict(weights_dict, strict=False)

    def fan_out(self):
        raise NotImplementedError

    def _fix_running_stats(self, layer, fix_params=False):

        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm):
            self.bn_frozen.append(layer)
            if fix_params and layer not in self.not_training:
                self.not_training.append(layer)
        elif isinstance(layer, list):
            for m in layer:
                self._fix_running_stats(m, fix_params)
        else:
            for m in layer.children():
                self._fix_running_stats(m, fix_params)

    def _fix_params(self, layer):

        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            self.not_training.append(layer)
        elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm):
            self.not_training.append(layer)
            self.bn_frozen.append(layer)
        elif isinstance(layer, list):
            for m in layer:
                self._fix_params(m)
        elif isinstance(layer, nn.Module):
            if hasattr(layer, "weight") or hasattr(layer, "bias"):
                print("Ignoring fixed weight/bias layer: ", layer)

            for m in layer.children():
                self._fix_params(m)

    def _freeze_bn(self, layer):

        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm):
            # freezing the layer
            layer.eval()
        elif isinstance(layer, nn.Module):
            for m in layer.children():
                self._freeze_bn(m)

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight.requires_grad = False

                if hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                print("Unkown layer to fix: ", layer)

        for bn_layer in self.bn_frozen:
            self._freeze_bn(bn_layer)

    def _lr_mult(self):
        return 1., 2., 10., 20

    def parameter_groups(self, base_lr, wd, **kwargs):

        w_old, b_old, w_new, b_new = self._lr_mult()

        groups = (
            {
                "params": [],
                "weight_decay": wd,
                "lr": w_old * base_lr
            },  # weight learning
            {
                "params": [],
                "weight_decay": 0.0,
                "lr": b_old * base_lr
            },  # bias finetuning
            {
                "params": [],
                "weight_decay": wd,
                "lr": w_new * base_lr
            },  # weight finetuning
            {
                "params": [],
                "weight_decay": 0.0,
                "lr": b_new * base_lr
            })  # bias learning

        for m in self.modules():

            if m in self.not_training:
                # skipping fixed layers
                continue

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d) or isinstance(
                    m, nn.SyncBatchNorm):

                if m.weight is not None:
                    if m in self.from_scratch_layers:
                        groups[2]["params"].append(m.weight)
                    else:
                        groups[0]["params"].append(m.weight)

                if m.bias is not None:
                    if m in self.from_scratch_layers:
                        groups[3]["params"].append(m.bias)
                    else:
                        groups[1]["params"].append(m.bias)

            elif hasattr(m, "weight"):
                print("! Skipping learnable: ", m)

        for i, g in enumerate(groups):
            print("Group {}: #{}, LR={:4.3e}".format(i, len(g["params"]), g["lr"]))

        return groups
