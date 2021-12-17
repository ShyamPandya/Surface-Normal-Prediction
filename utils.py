'''Functions for reading and saving EXR images using OpenEXR.
'''

import sys

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn as nn


def normal_to_rgb(normals_to_convert):
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) to be written
    into an image.
    The surface normals are normally in camera co-ords,
    with positive z axis coming out of the page. And the axes are
    mapped as (x,y,z) -> (R,G,B).
    Args:
        normals_to_convert (numpy.ndarray): Surface normals, dtype float32, range [-1, 1]
    '''
    camera_normal_rgb = (normals_to_convert + 1) / 2
    return camera_normal_rgb


def create_grid_image(inputs, outputs, labels, max_num_images_to_save=3):
    '''Make a grid of images for display purposes
    Size of grid is (3, N, 3), where each coloum belongs to input, output, label resp
    Args:
        inputs (Tensor): Batch Tensor of shape (B x C x H x W)
        outputs (Tensor): Batch Tensor of shape (B x C x H x W)
        labels (Tensor): Batch Tensor of shape (B x C x H x W)
        max_num_images_to_save (int, optional): Defaults to 3. Out of the given tensors, chooses a
            max number of imaged to put in grid
    Returns:
        numpy.ndarray: A numpy array with of input images arranged in a grid
    '''

    img_tensor = inputs[:max_num_images_to_save]

    output_tensor = outputs[:max_num_images_to_save]
    output_tensor_rgb = normal_to_rgb(output_tensor)

    label_tensor = labels[:max_num_images_to_save]

    label_tensor_rgb = normal_to_rgb(label_tensor)

    images = torch.cat((img_tensor, output_tensor_rgb, label_tensor_rgb), dim=3)
    # grid_image = make_grid(images, 1, normalize=True, scale_each=True)
    grid_image = make_grid(images, 1, normalize=False, scale_each=False)

    return grid_image


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter)**power)