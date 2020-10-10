import os
import mmcv
import cv2
import numpy as np

# test_image = '/home/aaron/Documents/datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_hair.png'
# image = mmcv.imread(test_image, 0)
# print(image.shape)
# print(np.max(image))
# image[np.where(image==255)] = 2
# print(np.max(image))

categories = ['_l_ear', '_r_ear', '_hair', '_skin', '_mouth', '_l_eye', '_r_eye', '_l_lip', '_u_lip', '_l_brow', '_r_brow', '_nose']
categories_dict = {'_l_ear': 1, 
                   '_r_ear': 1, 
                   '_hair': 2, 
                   '_skin': 3, 
                   '_mouth': 4, 
                   '_l_eye': 5, 
                   '_r_eye': 5, 
                   '_l_lip': 6, 
                   '_u_lip': 6, 
                   '_l_brow': 7, 
                   '_r_brow': 7, 
                   '_nose': 8}
label_path = '/home/aaron/Documents/datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno/'
# label_path = '/home/aaron/Documents/datasets/test_images/'
output_dir = '/home/aaron/Documents/datasets/face_seg/groundtruth/'
for label_sub_path in os.listdir(label_path):
    result_names = os.listdir(os.path.join(label_path, label_sub_path))
    for i in range(len(result_names)):
        result_names[i] = result_names[i].split('_')[0]
    result_names = sorted(list(set(result_names)))
    for result_name in result_names:
        result_image = np.zeros((512, 512))
        for category in categories:
            image_name = result_name + category + '.png'
            try:
                image = mmcv.imread(os.path.join(os.path.join(label_path, label_sub_path), image_name), 0)
                result_image[np.where(image == 255)] = categories_dict[category]
            except:
                continue
        cv2.imwrite(os.path.join(output_dir, result_name + '.png'), result_image)