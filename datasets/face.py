import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import mmcv
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms


class FaceParts(Dataset):

    def __init__(self, txt_file, image_path, gt_path, mode='train'):
        super().__init__()
        assert mode in ('train', 'val')
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        self.lines = lines
        self.image_path = image_path
        self.gt_path = gt_path
        self.mode = mode

    def __getitem__(self, index):
        image_name = self.lines[index] + '.jpg'
        image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        gt_name = '{:05d}.png'.format(int(self.lines[index]))
        gt = mmcv.imread(os.path.join(self.gt_path, gt_name), 0)
        num_classes = np.max(gt) + 1
        label = np.zeros((gt.shape[0], gt.shape[1], num_classes))
        for c in range(num_classes):
            label[:, :, c] = (gt[:, :] == c).astype(int)
        label = Image.fromarray(label)
        sample = {'image': image, 'label': label}
        if self.mode == 'train':
            sample = self.transform_train(sample)
        if self.mode == 'val':
            sample = self.transform_val(sample)
        return sample


    def __len__(self):
        return len(self.lines)

    def transform_train(self, sample):

        image_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomErasing(), 
            transforms.RandomGrayscale(),
            transforms.ColorJitter(brightness=(-0.5, .5), 
                                   contrast=(0, 0.5), 
                                   saturation=(0, 0.5), 
                                   hue=(-0.3, 0.3))
        ])
        composed_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomCrop((480, 480)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        ])
        sample['image'] = image_transforms(sample['image'])
        sample = composed_transforms(sample)
        return sample
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        sample = composed_transforms(sample)
        return sample