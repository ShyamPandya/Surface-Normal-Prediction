#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug as ia
from torch.utils.data import DataLoader
from torchvision import transforms
import h5py
import pickle
import os
import random


class SurfaceNormalsDataset(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations.
    If a label_dir is blank ( None, ''), it will assume labels do not exist and return a tensor of zeros
    for the label.
    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        label_dir (str): Path to folder containing the labels (.png format).
                         If no labels exists, pass empty string ('') or None.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img
    """

    def __init__(
            self,
            input_dir,
            label_dir,
            transform=None,
            input_only=None,
            is_train=True
    ):

        super().__init__()

        self.images_pickle = input_dir
        self.labels_pickle = label_dir
        self.transform = transform
        self.input_only = input_only
        self.is_train = is_train

        # Create list of filenames
        self.images_list = []
        self.labels_list = []
        self.datalist_input = []
        self.labels_input = []
        self.read_data()


    def __len__(self):
        return len(self.datalist_input)

    def read_data(self):
        limit = 40000 if self.is_train else 5000
        #limit = 500 if self.is_train else 100

        with open(self.images_pickle, 'rb') as file:
            image_list = pickle.load(file)
            file.close()

        with open(self.labels_pickle, 'rb') as file:
            label_list = pickle.load(file)
            file.close()

        temp = list(zip(image_list, label_list))
        random.seed(69)
        random.shuffle(temp)
        image_list, label_list = zip(*temp)

        if self.is_train:
            image_list = image_list[:40000]
            label_list = label_list[:40000]
        else:
            image_list = image_list[-8000:]
            label_list = label_list[-8000:]

        counter = 0
        for i in range(len(image_list)):
            if counter == limit:
                break
            image_name = image_list[i]
            label_name = label_list[i]
            self.datalist_input.append(image_name)
            self.labels_input.append(label_name)
            counter += 1


    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.
        Args:
            index (int): index of the item required from dataset.
        Returns:
            torch.Tensor: Tensor of input image
            torch.Tensor: Tensor of label (Tensor of zeroes is labels_dir is "" or None)
        '''

        # Open input imgs
        _img_path = self.datalist_input[index]
        _label_path = self.labels_input[index]

        hdf_image = h5py.File('../chupao/jpegs/' + _img_path, 'r')
        hdf_label = h5py.File('../chupao/normals-r/' + _label_path, 'r')

        _img = hdf_image['dataset'][:].astype('float32')
        _label = hdf_label['dataset'][:].astype('float32')

        #print(_img)

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()

            _img = det_tf.augment_image(_img)

            # Making all values of invalid pixels marked as -1.0 to 0.
            # In raw data, invalid pixels are marked as (-1, -1, -1) so that on conversion to RGB they appear black.
            mask = np.all(_label == -1.0, axis=0)
            _label[:, mask] = 0.0

            #_label = _label.transpose((1, 2, 0))  # To Shape: (H, W, 3)
            _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))

        _label = _label.transpose((2, 0, 1))  # To Shape: (3, H, W)

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)

        _label_tensor = torch.from_numpy(_label)
        # _label_tensor = nn.functional.normalize(_label_tensor, p=2, dim=0)

        return _img_tensor, _label_tensor

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default
