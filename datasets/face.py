import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import mmcv
import numpy as np
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


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
        image_name = self.lines[index].strip('\n')[0] + '.jpg'
        image = Image.open(os.path.join(self.image_path, image_name))

        gt_name = '{:05d}.png'.format(int(self.lines[index].strip('\n')[0] ))
        gt = mmcv.imread(os.path.join(self.gt_path, gt_name), 0)

        # num_classes = np.max(gt) + 1
        # label = np.zeros((gt.shape[0], gt.shape[1], num_classes))
        # for c in range(num_classes):
        #     label[:, :, c] = (gt[:, :] == c).astype(int)
        # label = Image.fromarray(label)
        label = Image.fromarray(gt)
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
            transforms.RandomGrayscale(),
            transforms.ColorJitter(brightness=0.5, 
                                   contrast=0.5, 
                                   saturation=0.5, 
                                   ),
        ])
        p = random.randint(0, 1)
        composed_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomCrop((480, 480)),
            transforms.RandomHorizontalFlip(p),
            transforms.ToTensor(),
            

        ])
        normalize_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        sample['image'] = image_transforms(sample['image'])
        
        seed = np.random.randint(2020)
        random.seed(seed)
        sample['image'] = composed_transforms(sample['image'])
        sample['label'] = composed_transforms(sample['label']) * 255
        sample['label'] = sample['label'].int()
        sample['image'] = normalize_transform(sample['image'])

        return sample
    def transform_val(self, sample):
        image_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        composed_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            
        ])
        
        sample['image'] = composed_transforms(sample['image'])
        sample['label'] = composed_transforms(sample['label'])
        sample['image'] = image_transforms(sample['image'])
        return sample

if __name__ == "__main__":

    train_txt = '/home/aaron/Documents/datasets/face_seg/train_test/train.txt'
    val_txt = '/home/aaron/Documents/datasets/face_seg/train_test/test.txt'
    image_path = '/home/aaron/Documents/datasets/face_seg/CelebA-HQ-img'
    gt_path = '/home/aaron/Documents/datasets/face_seg/groundtruth'

    data = FaceParts(train_txt, 
              image_path, 
              gt_path,
              mode='train'
              )[0]
    image = np.array(data['image'])
    image = np.transpose(image, [1, 2, 0])
    plt.figure('image')
    plt.imshow(image)


    label = np.array(data['label'])[0, :, :]
    print(np.max(label))
    plt.figure('test')
    plt.imshow(label)
    plt.show()