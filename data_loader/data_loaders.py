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

# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# FingerCountingDataLoader
class FingerCountingDataLoader(BaseDataLoader):
    """
    FingerCountingDataLoader for loading finger counting dataset
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm =  T.Compose([
            T.ToTensor(),
            # T.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),            
        ])

        self.data_dir = data_dir
        split = 'train' if training else 'test'

        # dataset 
        self.dataset = FingerCountDataset(self.data_dir, split=split, transform=trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)