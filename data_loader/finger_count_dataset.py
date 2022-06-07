import torch
import nibabel as nib
from torch.utils import data
import os
import numpy as np
import torch.nn.functional as F
import cv2
import xmltodict, json
import pprint

# from transformations.transformations import normalize_01
import torchvision.transforms as transforms
from PIL import Image

from time import sleep

# digits lookup table
digits = {
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
        self.data_dir_img = os.path.join(data_dir, split)
        self.transform = transform

        # list all files in directory
        files = os.listdir(self.data_dir_img)
        n_files = len(files)
        n_images = n_files // 2

        # images
        files_imgs = list(sorted([f for f in files if f.endswith('.png')]))
        print('Found {} img files'.format(len(files_imgs)))

        # load images
        self.inputs = []
        self.targets = []  # is a list [batch_size][N][[x0, y0, x1, y1]]

        for idx, filename in enumerate(files_imgs):
            # print('Loading image {}/{}'.format(idx+1, len(files_imgs)))
            # img = cv2.imread(os.path.join(self.data_dir_img, filename))
            img = Image.open(os.path.join(self.data_dir_img, filename)).convert("RGB")
            self.inputs.append(img)

        # load label, get bounding box coordinates for each mask
        files_xml = list(sorted([f for f in files if f.endswith('.xml')]))
        print('Found {} xml files'.format(len(files_xml)))

        for file_idx, filename in enumerate(files_xml):
            obj = xmltodict.parse(open(self.data_dir_img + '/' + filename).read())

            # get the annotation for the current image
            annotations = obj['annotation']['object']
            
            # if there is only one object, wrap in list
            if not isinstance(annotations, list):
                annotations = [annotations]

            annotate_list = []

            # get bounding box coordinates for each annotation
            for a_idx, annotation in enumerate(annotations):
                xmin = int(annotation['bndbox']['xmin'])
                ymin = int(annotation['bndbox']['ymin'])
                xmax = int(annotation['bndbox']['xmax'])
                ymax = int(annotation['bndbox']['ymax'])
                class_name = annotation['name']

                # create list [N][[x0, y0, x1, y1]] for each annotation 
                annotate_list.append([int(digits[class_name]), xmin, ymin, xmax, ymax])


            # append to targets
            self.targets.append(annotate_list)

            # print self.targets for testing
            # print("len targets: {}".format(len(self.targets)))
            # print("targets[file_idx]: {}".format(self.targets[file_idx]))
            # print("len targets[0][0]: {}".format(len(self.targets[0][0])))
        
        # print len of inputs and targets
        print("len inputs: {}".format(len(self.inputs)))
        print("len targets: {}".format(len(self.targets)))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):    
        # see: https://pytorch.org/vision/stable/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html   

        ### image: a PIL Image of size (H, W) ###
        # The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each image, and should be in 0-1 range. 
        # Different images can have different sizes.
        img = self.inputs[index]
        
        ### target: a dict containing the following fields ###
        # boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
        # labels (Int64Tensor[N]): the class label for each bounding box
        # image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
        # target_area (Tensor[N]): the area of the bounding box. The area is calculated using `x1 - x0` and `y1 - y0`
        # iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.

        target_info = self.targets[index] # target shape = [int(digits[class_name]), xmin, ymin, xmax, ymax]

        boxes = []
        labels = []
        image_id = []
        area = []
        iscrowd = []

        for target in target_info:
            # get info
            label = target[0]
            xmin = target[1]
            ymin = target[2]
            xmax = target[3]
            ymax = target[4]
            area = (xmax - xmin) * (ymax - ymin)
            # print("area: {}".format(area))

            # write to list
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            iscrowd.append(0)

        # convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = torch.as_tensor(area, dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int8)

        # construct dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img = self.transform(img)

        # img = transforms.ToTensor(img.ToTensor())

        # return x, y
        return img, target
