import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import mmcv
import cv2
import numpy as np
import sys
sys.path.append('./')
import models.unet as unet
from datasets.face import FaceParts


# load model
model = unet.Unet()
model = model.cuda()
# print(model)
cudnn.benchmark = True
inputs = torch.rand(2, 3, 480, 480)
inputs = inputs.cuda()
outputs = model(inputs)
print(outputs.size())


# load data

train_txt = '/home/aaron/Documents/datasets/face_seg/train_test/train.txt'
val_txt = '/home/aaron/Documents/datasets/face_seg/train_test/test.txt'
image_path = '/home/aaron/Documents/datasets/face_seg/CelebA-HQ-img'
gt_path = '/home/aaron/Documents/datasets/face_seg/groundtruth'

train_loader = DataLoader(
    FaceParts(train_txt, 
              image_path, 
              gt_path,
              mode='train'
              ),
    batch_size=4,
    shuffle=True
    )
val_loader = DataLoader(
    FaceParts(val_txt, 
              image_path, 
              gt_path,
              mode='val'
              ),
    batch_size=1,
    shuffle=False
    )
print(len(train_loader))
print(len(val_loader))

