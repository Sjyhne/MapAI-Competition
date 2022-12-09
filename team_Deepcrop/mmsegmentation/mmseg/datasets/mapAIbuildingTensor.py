# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class NORAmapAITensorDataset(CustomDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('building', 'other')
    # PALETTE = [[0, 0, 0], [255, 255, 255]]
    PALETTE = [ [255, 255, 255], [0, 0, 0]]

    # CLASSES = 'building'
    # PALETTE = [[255, 255, 255]]

    def __init__(self, **kwargs):
        super(NORAmapAITensorDataset, self).__init__(
            img_suffix='.pt',
            seg_map_suffix='.tif',
            reduce_zero_label=False, ## if ignore the background class
            **kwargs)
