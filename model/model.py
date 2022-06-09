import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FingerDetector(BaseModel):
    def __init__(self):
        super().__init__()

        # import pretrained faster rcnn model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # self.model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)

        # replace the classifier for 5 fingers + background = 6 classes
        self.num_classes = 6 

        # get number of input features for the classifier
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes) 

    def forward(self, images, targets):
        return self.model(images, targets)