import os
import os.path as osp
import sys

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage import io

sys.path.append('./')

from sembm.core.opts import get_arguments  # noqa
from sembm.core.config import cfg, cfg_from_file, cfg_from_list  # noqa
from sembm.datasets import build_dataset  # noqa

if __name__ == '__main__':

    # loading the model
    args = get_arguments(sys.argv[1:])

    # reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    dataset = build_dataset(cfg, 'val')

    for img_path in dataset.images:
        filename = osp.splitext(osp.basename(img_path))[0]
        img = io.imread(img_path)
        segments = slic(img, n_segments=60, compactness=10)
        out = mark_boundaries(img, segments)
        # print(segments)
        plt.subplot(121)
        plt.title("n_segments=60")
        plt.imshow(out)

        segments2 = slic(img, n_segments=300, compactness=10)
        print(segments2.shape)
        print(np.unique(segments2))
        exit(0)
        out2 = mark_boundaries(img, segments2)
        plt.subplot(122)
        plt.title("n_segments=300")
        plt.imshow(out2)

        plt.savefig(f'show_spixels/{filename}.png')
