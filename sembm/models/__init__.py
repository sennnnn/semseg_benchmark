# from functools import partial
# from .stage_net import network_factory
from .ae import AE
from .simple_bsl import SimpleBaseline
from .mctformer import deit_small_MCTformerV2_patch16_224
from .deeplabv1 import Deeplabv1


def build_model(cfg):
    if cfg.NET.MODEL == 'ae':
        return AE(cfg.NET.BACKBONE, cfg.NET.PRE_WEIGHTS_PATH, cfg.NET.BN_TYPE, cfg.NET.NUM_CLASSES)
    elif cfg.NET.MODEL == 'simple_bsl':
        return SimpleBaseline(cfg.NET.BACKBONE, cfg.NET.PRE_WEIGHTS_PATH, cfg.NET.BN_TYPE, cfg.NET.NUM_CLASSES)
    elif cfg.NET.MODEL == 'mctformer':
        return deit_small_MCTformerV2_patch16_224(
            backbone_name=cfg.NET.BACKBONE,
            pre_weights_path=cfg.NET.PRE_WEIGHTS_PATH,
            norm_type=cfg.NET.BN_TYPE,
            num_classes=cfg.NET.NUM_CLASSES,
            pretrained=True,
            drop_rate=0.0,
            drop_path_rate=0.1)
    elif cfg.NET.MODEL == 'deeplabv1':
        return Deeplabv1(cfg.NET.BACKBONE, cfg.NET.PRE_WEIGHTS_PATH, cfg.NET.BN_TYPE, cfg.NET.NUM_CLASSES)
