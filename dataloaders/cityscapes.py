#!/usr/bin/env python3
"""
Created on Sat Oct 20 00:18:43 2018

@author: Taha Emara  @email: taha@emaraic.com
"""

import os
import numpy as np
from PIL import Image
from torch.utils import data

from dataloaders.utils import listFiles

class Cityscapes(data.Dataset):

    def __init__(self, root='path/to/datasets/cityscapes', split="train", transform=None,extra=False):
        """
        Cityscapes dataset folder has two folders, 'leftImg8bit' folder for images and 'gtFine_trainvaltest' 
        folder for annotated images with fine annotations 'labels'.
        """
        self.root = root
        self.split = split #train, validation, and test sets
        self.transform = transform
        self.files = {}
        self.n_classes = 19
        self.extra=extra

        if not self.extra:
            print("Using fine dataset")
            self.images_path = os.path.join(self.root, 'leftImg8bit_trainvaltest','leftImg8bit', self.split)
            self.labels_path = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)
        else:
            print("Using Coarse dataset")

            self.images_path = os.path.join(self.root, 'leftImg8bit', self.split)
            self.labels_path = os.path.join(self.root, 'gtCoarse', 'gtCoarse', self.split)            
            
        #print(self.images_path)
        self.files[split] = listFiles(rootdir=self.images_path, suffix='.png')#list of the pathes to images

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1] #not to train
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))
        #print(self.class_map)
        
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images.path))

        print("Found %d %s images" % (len(self.files[split]), split))
        
    
    def __len__(self):
        return len(self.files[self.split])
    
    def __getitem__(self, index):
        image_path = self.files[self.split][index].rstrip()
        #print(image_path)
        if not self.extra:
            label_path = os.path.join(self.labels_path,
                                image_path.split(os.sep)[-2],
                                os.path.basename(image_path)[:-15] + 'gtFine_labelIds.png')
        else:
            label_path = os.path.join(self.labels_path,
                                image_path.split(os.sep)[-2],
                                os.path.basename(image_path)[:-15] + 'gtCoarse_labelIds.png')
        _img = Image.open(image_path).convert('RGB')
        _tmp = np.array(Image.open(label_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)

        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def encode_segmap(self, mask):
        # Put all void classes to ignore_index
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    
