# Surface-Normal-Prediction

## Participants
Shyam Pandya - sp3912 <br>
Harjot Singh - hs3263 <br>
Ishaan Vijh - iv2245
 
## Background

Surface normals are unit vectors perpendicular to the surface at any given spot. The rate at which they change can help define the surface’s curvature. They also help describe the orientation of the surface (the direction that it 'faces'). A surface normal is a vector value of the form (dx, dy, dz). 

## Project Description

Surface normals are very widely used for 3D shape modeling of real-world objects. However, it is difficult to find data that provides pixel-wise surface normals and thus our target for this project is to build a novel deep learning based approach to estimate pixel-wise surface normal vectors for a given RGB image.


## Dataset

Apple recently released a photorealistic synthetic dataset for indoor scenes called Hyperism [1] in ICCV 2021. They have 77,400 images of resolution 1024 × 768 for 461 indoor scenes with detailed per-pixel labels and corresponding ground truth geometry. The entire dataset is roughly 1.9TB with multiple features like depth, position, surface normal, illumination etc. This data is split into train/val/test by randomly partitioning the data at the granularity of scenes, rather than images or camera trajectories, to minimize the probability of very similar images ending up in different partitions. As the size of this dataset is very large, we will further choose a random set of indoor scenes from each split and build a model with 10,000 / 2,000 / 2,000 train/val/test images. We will also try to gather some real world images using an RGBD camera to test our model and even tra

## Framework and architecture

We will be using the PyTorch framework in order to build models for this project as it is very easy to write and has a very active developing community and good documentation. We can very easily use model APIs in pytorch and even obtain pretrained weights for SOTA vision backbones like ResNet, VGG16, EfficientNet [2] etc.
We will be using ResNet-50 as a backbone for the surface normal prediction task. ResNet allows us to train extremely deep neural networks with 150+layers successfully. Training very deep neural networks was difficult prior to ResNet due to the problem of vanishing gradients. ResNet uses what is known as shortcut/skip connection to flow data easily without hampering the learning ability of deep learning models. The advantage of adding this type of skip connection is that if any layer hurts the performance of the model, it will be skipped.
We will be using transfer learning on the ResNet backbone by freezing the early layers and replacing the classification head with a surface normal predictor for each pixel and then fine-tuning the weights of the model after unfreezing the early layers. The input would be a downsampled and transformed image from the dataset and the output would be the pixel-wise normals for this downsampled image. We will be building a custom loss function similar to the one mentioned in [3] to estimate the correctness of the predicted values.
If time permits, we will be experimenting with other SOTA architures and a custom neural network built specifically for surface normal prediction using [3] as a reference.

## References

[1] https://machinelearning.apple.com/research/hypersim <br>
[2] https://pytorch.org/vision/stable/models.html <br>
[3] X.Wang,D.Fouhey,and A.Gupta.Designing Deep Networks for surface normal estimation. In CVPR, 2015. <br>
[4] Floors are Flat: Leveraging Semantics for Real-Time Surface Normal Prediction 


