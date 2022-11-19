# from functools import partial
# from .stage_net import network_factory
from sembm.models.deeplabv1_cls import Deeplabv1Cls
from .ae import AE
from .simple_bsl import SimpleBaseline
from .mctformer import deit_small_MCTformerV2_patch16_224
from .deeplabv1 import Deeplabv1
from .deeplabv1_neg import Deeplabv1Neg
from .deeplabv1_cls import Deeplabv1Cls
from .deeplabv1_seam import Deeplabv1SEAM
from .deeplabv1_can import Deeplabv1CAN
from .deeplabv1_seam_can import Deeplabv1SEAMCAN
from .deeplabv1_dual import Deeplabv1SEAMDual
from .deeplabv1_seam_hardness import Deeplabv1SEAMHardness


def build_model(cfg):
    if cfg.NET.MODEL == '1-stage-wsss':
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
    elif cfg.NET.MODEL == 'deeplabv1_neg':
        return Deeplabv1Neg(cfg.NET.BACKBONE, cfg.NET.PRE_WEIGHTS_PATH, cfg.NET.BN_TYPE, cfg.NET.NUM_CLASSES)
    elif cfg.NET.MODEL == 'deeplabv1_cls':
        return Deeplabv1Cls(cfg.NET.BACKBONE, cfg.NET.PRE_WEIGHTS_PATH, cfg.NET.BN_TYPE, cfg.NET.NUM_CLASSES)
    elif cfg.NET.MODEL == 'deeplabv1_seam':
        return Deeplabv1SEAM(cfg.NET.BACKBONE, cfg.NET.PRE_WEIGHTS_PATH, cfg.NET.BN_TYPE, cfg.NET.NUM_CLASSES)
    elif cfg.NET.MODEL == 'deeplabv1_can':
        return Deeplabv1CAN(cfg.NET.BACKBONE, cfg.NET.PRE_WEIGHTS_PATH, cfg.NET.BN_TYPE, cfg.NET.NUM_CLASSES)
    elif cfg.NET.MODEL == 'deeplabv1_seam_can':
        return Deeplabv1SEAMCAN(cfg.NET.BACKBONE, cfg.NET.PRE_WEIGHTS_PATH, cfg.NET.BN_TYPE, cfg.NET.NUM_CLASSES, cfg.NET.ALPHA, cfg.NET.MARGIN)
    elif cfg.NET.MODEL == 'deeplabv1_dual':
        return Deeplabv1SEAMDual(cfg.NET.BACKBONE, cfg.NET.PRE_WEIGHTS_PATH, cfg.NET.BN_TYPE, cfg.NET.NUM_CLASSES)
    elif cfg.NET.MODEL == 'deeplabv1_seam_hardness':
        return Deeplabv1SEAMHardness(cfg.NET.BACKBONE, cfg.NET.PRE_WEIGHTS_PATH, cfg.NET.BN_TYPE, cfg.NET.NUM_CLASSES)
