import torch.nn as nn
import torch

import math


def loss_fn_cosine(input_vec, target_vec, reduction='sum'):
    '''A cosine loss function for use with surface normals estimation.
    Calculates the cosine loss between 2 vectors. Both should be of the same size.
    Arguments:
        input_vec {tensor} -- The 1st vectors with whom cosine loss is to be calculated
                              The dimensions of the matrices are expected to be (batchSize, 3, height, width).
        target_vec {tensor } -- The 2nd vectors with whom cosine loss is to be calculated
                                The dimensions of the matrices are expected to be (batchSize, 3, height, width).
    Keyword Arguments:
        reduction {str} -- Can have values 'elementwise_mean' and 'none'.
                           If 'elemtwise_mean' is passed, the mean of all elements is returned
                           if 'none' is passed, a matrix of all cosine losses is returned, same size as input.
                           (default: {'elementwise_mean'})
    Raises:
        Exception -- Exception is an invalid reduction is passed
    Returns:
        tensor -- A single mean value of cosine loss or a matrix of elementwise cosine loss.
    '''
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = 1.0 - cos(input_vec, target_vec)

    # calculate loss only on valid pixels
    # mask_invalid_pixels = (target_vec[:, 0, :, :] == -1.0) & (target_vec[:, 1, :, :] == -1.0) & (target_vec[:, 2, :, :] == -1.0)
    mask_invalid_pixels = torch.all(target_vec == -1, dim=1) & torch.all(target_vec == 0, dim=1)

    loss_cos[mask_invalid_pixels] = 0.0
    loss_cos_sum = loss_cos.sum()
    total_valid_pixels = (~mask_invalid_pixels).sum()
    error_output = loss_cos_sum / total_valid_pixels

    if reduction == 'elementwise_mean':
        loss_cos = error_output
    elif reduction == 'sum':
        loss_cos = loss_cos_sum
    elif reduction == 'none':
        loss_cos = loss_cos
    else:
        raise Exception(
            'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())

    return loss_cos


def metric_calculator_batch(input_vec, target_vec, mask=None):
    """Calculate mean, median and angle error between prediction and ground truth
    Args:
        input_vec (tensor): The 1st vectors with whom cosine loss is to be calculated
                            The dimensions of are expected to be (batchSize, 3, height, width).
        target_vec (tensor): The 2nd vectors with whom cosine loss is to be calculated.
                             This should be GROUND TRUTH vector.
                             The dimensions are expected to be (batchSize, 3, height, width).
        mask (tensor): The pixels over which loss is to be calculated. Represents VALID pixels.
                             The dimensions are expected to be (batchSize, 3, height, width).
    Returns:
        float: The mean error in 2 surface normals in degrees
        float: The median error in 2 surface normals in degrees
        float: The percentage of pixels with error less than 11.25 degrees
        float: The percentage of pixels with error less than 22.5 degrees
        float: The percentage of pixels with error less than 3 degrees
    """
    if len(input_vec.shape) != 4:
        raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(input_vec.shape))
    if len(target_vec.shape) != 4:
        raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(target_vec.shape))

    INVALID_PIXEL_VALUE = 0  # All 3 channels should have this value
    mask_valid_pixels = ~(torch.all(target_vec == INVALID_PIXEL_VALUE, dim=1))
    if mask is not None:
        mask_valid_pixels = (mask_valid_pixels.float() * mask).byte()
    total_valid_pixels = mask_valid_pixels.sum()
    if (total_valid_pixels == 0):
        print('[WARN]: Image found with ZERO valid pixels to calc metrics')
        return torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), mask_valid_pixels

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = cos(input_vec, target_vec)

    # Taking torch.acos() of 1 or -1 results in NaN. We avoid this by small value epsilon.
    eps = 1e-10
    loss_cos = torch.clamp(loss_cos, (-1.0 + eps), (1.0 - eps))
    loss_rad = torch.acos(loss_cos)
    loss_deg = loss_rad * (180.0 / math.pi)

    # Mask out all invalid pixels and calc mean, median
    loss_deg = loss_deg[mask_valid_pixels]
    loss_deg_mean = loss_deg.mean()
    loss_deg_median = loss_deg.median()

    # Calculate percentage of vectors less than 11.25, 22.5, 30 degrees
    percentage_1 = ((loss_deg < 11.25).sum().float() / total_valid_pixels) * 100
    percentage_2 = ((loss_deg < 22.5).sum().float() / total_valid_pixels) * 100
    percentage_3 = ((loss_deg < 30).sum().float() / total_valid_pixels) * 100

    return loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3