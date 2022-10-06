from torch.utils import data
from .pascal_voc import VOCSegmentation
from .pascal_voc import VOCSegmentation_Augmentation
from .pascal_voc_pseudo_gt import VOCSegmentationPseudoGT
from .voc_pseudo_gt import VOCPseudoGT

datasets = {
    'pascal_voc': VOCSegmentation,
    'pascal_voc_daug': VOCSegmentation_Augmentation,
    'pascal_voc_pseudo_gt': VOCSegmentationPseudoGT,
    'voc_pseudo_gt': VOCPseudoGT,
}


def get_num_classes(dataset_name):
    return datasets[dataset_name.lower()].NUM_CLASSES


def get_class_names(dataset_name):
    return datasets[dataset_name.lower()].CLASSES


def get_dataloader(dataset_name, cfg, split, batch_size, num_workers, test_mode=False):
    assert split in ('train', 'train_voc', 'val'), "Unknown split '{}'".format(split)

    dataset_name = dataset_name.lower()
    dataset_cls = datasets[dataset_name]
    dataset = dataset_cls(cfg, split)

    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    shuffle, drop_last = [True, True] if split == 'train' else [False, False]

    return data.DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, **kwargs)


def build_dataset(cfg, split):
    dataset_name = cfg.DATASET.NAME
    if dataset_name == 'pascal_voc':
        return VOCSegmentation(cfg, split)
    elif dataset_name == 'pascal_voc_daug':
        return VOCSegmentation_Augmentation(cfg, split)
    elif dataset_name == 'pascal_voc_pseudo_gt':
        return VOCSegmentationPseudoGT(cfg, split)
    elif dataset_name == 'voc_pseudo_gt':
        return VOCPseudoGT(cfg, split)
