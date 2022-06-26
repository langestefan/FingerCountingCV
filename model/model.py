import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

#------- FCOS FineTuning -------
def FCOS_FineTuning(num_classes, pretrained_bb=True, trainable_layers=4):

    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True, 
                                                           pretrained_backbone=pretrained_bb,
                                                           trainable_backbone_layers=trainable_layers)
    
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    out_channels = 256

    cls_logits = nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
    nn.init.normal_(cls_logits.weight, std=0.01)
    nn.init.constant_(cls_logits.bias, -np.log((1 - 0.01) / 0.01))

    model.head.classification_head.cls_logits = cls_logits

    return model

#------- FCOS FromScratch -------
def FCOS_FromScratch(num_classes, pretrained_bb=True, trainable_layers=4):

    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=False, 
                                                           pretrained_backbone=pretrained_bb,
                                                           trainable_backbone_layers=trainable_layers)
    
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    out_channels = 256

    cls_logits = nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
    nn.init.normal_(cls_logits.weight, std=0.01)
    nn.init.constant_(cls_logits.bias, -np.log((1 - 0.01) / 0.01))

    model.head.classification_head.cls_logits = cls_logits

    return model

class FingerDetector(BaseModel):
    def __init__(self):
        super().__init__()

        # replace the classifier for 5 fingers + background = 6 classes
        self.num_classes = 6 

        # import FCOS resnet model
        self.model = FCOS_FineTuning(self.num_classes)
        # self.model = FCOS_FromScratch(self.num_classes)

    def forward(self, images, targets):
        return self.model(images, targets)