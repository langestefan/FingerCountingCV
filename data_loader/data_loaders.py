from torchvision import datasets
from torchvision import transforms as T
from base import BaseDataLoader

from data_loader.finger_count_dataset import FingerCountDataset

from utils import inf_loop, MetricTracker, collate_fn

import albumentations as A
from albumentations.pytorch import ToTensorV2

# FingerCountingDataLoader
class FingerCountingDataLoader(BaseDataLoader):
    """
    FingerCountingDataLoader for loading finger counting dataset
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        transform = A.Compose([A.HorizontalFlip(p=0.5),
                               A.ShiftScaleRotate(p=0.5, scale_limit=0.3),
                               A.RandomBrightnessContrast(p=0.3),
                               A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
                               A.AdvancedBlur(p=0.5),
                               ToTensorV2()],
                	           bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)

        self.data_dir = data_dir
        split = 'train' if training else 'test'

        # dataset 
        self.dataset = FingerCountDataset(self.data_dir, split=split, transform=transform)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)