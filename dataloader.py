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
    ):

        super().__init__()

        self.images_pickle = input_dir
        self.labels_pickle = label_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self.images_list = []
        self.labels_list = []
        self.datalist_input = []
        self.labels_input = []
        self.read_data()


    def __len__(self):
        return min(len(self.datalist_input), len(self.labels_input))

    def read_data(self):
        with open(self.images_pickle, 'rb') as file:
            temp = pickle.load(file)
        for i in temp:
            if os.path.exists('../hdf5s-r/' + i):
                self.datalist_input.append(i)
        with open(self.labels_pickle, 'rb') as file1:
            temp = pickle.load(file1)
        for i in temp:
            if os.path.exists('../normals-r/' + i):
                self.labels_input.append(i)

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

        hdf_image = h5py.File('../hdf5s-r/' + _img_path, 'r')
        hdf_label = h5py.File('../normals-r/' + _label_path, 'r')

        _img = hdf_image['dataset'][:].astype('uint8')
        _label = hdf_label['dataset'][:].astype('uint8')

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()

            _img = det_tf.augment_image(_img)
            #_img = _img.transpose((2, 0, 1))

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


    # Example Augmentations using imgaug
    # imsize = 512
    # augs_train = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0), # Resize image
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.Rot90((0, 4)),
    #     # Blur and Noise
    #     #iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
    #     #iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
    #     iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),
    #     # Color, Contrast, etc.
    #     #iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
    #     iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
    #     iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
    #     #iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    # ])
    # augs_test = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0),
    # ])

'''augs = None  # augs_train
input_only = None  # ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

db_test = SurfaceNormalsDataset(input_dir="C:/Users/Classified/Documents/Surface Normals/Surface-Normal-Prediction/hdf5.pickle",
                                label_dir="C:/Users/Classified/Documents/Surface Normals/Surface-Normal-Prediction/normal.pickle",
                                transform=augs,
                                input_only=input_only)

batch_size = 16
testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=32, drop_last=True)'''

