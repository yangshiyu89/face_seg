import os
image_path = '/home/aaron/Documents/datasets/face_seg/CelebA-HQ-img'
gt_path = '/home/aaron/Documents/datasets/face_seg/groundtruth'
train_test_path = '/home/aaron/Documents/datasets/face_seg/train_test'
len_files = len(os.listdir(image_path))
TRAIN_RATIO = 0.8
len_train = round(len_files * TRAIN_RATIO)
print(len_files, len_train)
len_test = len_files - len_train
index = 0
train_f = open(train_test_path + '/train.txt', 'w')
test_f = open(train_test_path + '/test.txt', 'w')

for i, image in enumerate(sorted(os.listdir(image_path))):
    gt = image.split('.')[0]
    gt = '{:05d}.png'.format(int(gt))
    if gt in sorted(os.listdir(gt_path)) and i < len_train:
        train_f.write(image.split('.')[0] + '\n')
    if gt in sorted(os.listdir(gt_path)) and i >= len_train:
        test_f.write(image.split('.')[0] + '\n')
    if i % 100 == 0:
        print(i)
train_f.close()
test_f.close()