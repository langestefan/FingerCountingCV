import torch
import nibabel as nib
from torch.utils import data
import os
import numpy as np
import torch.nn.functional as F
import cv2
import xmltodict, json

# from transformations.transformations import normalize_01
import torchvision.transforms as transforms

from time import sleep

# digits lookup table
digits = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9
}


class FingerCountDataset(data.Dataset):
    def __init__(self,          
                data_dir: str,
                split: str='train',
                transform=None
                ):

        assert split in ['train', 'test'], 'Only train, test are supported options.'
        assert os.path.exists(data_dir), 'data_dir path does not exist: {}'.format(data_dir)

        print('Loading dataset for {}'.format(data_dir+'/'+split+'/'))

        self.data_dir = data_dir
        self.split = split

        # list all files in directory
        files = os.listdir(data_dir)
        n_files = len(files)
        n_images = n_files // 2

        # images
        files_imgs = list(sorted([f for f in files if f.endswith('.png')]))
        print('Found {} img files'.format(len(files_imgs)))

        # load images
        self.inputs = []
        self.targets = []

        for idx, filename in enumerate(files_imgs):
            print('Loading image {}/{}'.format(idx+1, len(files_imgs)))
            img = cv2.imread(os.path.join(data_dir, filename))
            self.inputs.append(img)

        # load label, get bounding box coordinates for each mask
        files_xml = list(sorted([f for f in files if f.endswith('.xml')]))
        print('Found {} xml files'.format(len(files_xml)))

        for file_idx, filename in enumerate(files_xml):
            obj = xmltodict.parse(open(self.data_dir + '/' + filename).read())
            print('Loading xml {}/{}'.format(idx+1, len(files_xml)))

            # get the annotation for the current image
            annotations = obj['annotation']['object']
            
            # if there is only one object, wrap in list
            if not isinstance(annotations, list):
                annotations = [annotations]

            # get bounding box coordinates for each annotation
            for annotation in annotations:
                print("At filename: ", filename)
                xmin = int(annotation['bndbox']['xmin'])
                ymin = int(annotation['bndbox']['ymin'])
                xmax = int(annotation['bndbox']['xmax'])
                ymax = int(annotation['bndbox']['ymax'])
                class_name = annotation['name']
                print("class_name: ", class_name)

                # create dictionary for each annotation and append to current target index
                target = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'digit': int(digits[class_name])}

                # target belongs to current image, so we must append to current image
                self.targets[file_idx].append(target)











    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):       

        x, y = 0, 0


        # if self.transform is not None:
        #     x, y = self.transform(x, y)        

        # typecasting    
        x, y = x.type(self.inputs_dtype), y.type(self.targets_dtype)

        return x, y