from torchvision import datasets
from torchvision import transforms as T
from base import BaseDataLoader

from data_loader.finger_count_dataset import FingerCountDataset

from utils import inf_loop, MetricTracker, collate_fn


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# FingerCountingDataLoader
class FingerCountingDataLoader(BaseDataLoader):
    """
    FingerCountingDataLoader for loading finger counting dataset
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm =  T.Compose([
            T.ToTensor()
        ])

        self.data_dir = data_dir
        split = 'train' if training else 'test'

        # dataset 
        self.dataset = FingerCountDataset(self.data_dir, split=split, transform=trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)