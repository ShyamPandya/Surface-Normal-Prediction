import os

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import torch.nn as nn
import torch

import loss_functions
from dataloader import SurfaceNormalsDataset
from model import SurfaceNormalModel
from utils import create_grid_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log_dir = "./runs/"
writer = SummaryWriter(log_dir, comment='create-graph')
'''checkpoint_dir = log_dir + "/checkpoints"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

input_only = [
    "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
    "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
]

augs_train = iaa.Sequential([
    # Geometric Augs
    # iaa.Resize({
    #    "height": 256,
    #    "width": 256
    # }, interpolation='nearest'),
    # iaa.Fliplr(0.5),
    # iaa.Flipud(0.5),
    # iaa.Rot90((0, 4)),

    # Bright Patches
    iaa.Sometimes(
        0.1,
        iaa.blend.Alpha(factor=(0.2, 0.7),
                        first=iaa.blend.SimplexNoiseAlpha(first=iaa.Multiply((1.5, 3.0), per_channel=False),
                                                          upscale_method='cubic',
                                                          iterations=(1, 2)),
                        name="simplex-blend")),

    # Color Space Mods
    iaa.Sometimes(
        0.3,
        iaa.OneOf([
            iaa.Add((20, 20), per_channel=0.7, name="add"),
            iaa.Multiply((1.3, 1.3), per_channel=0.7, name="mul"),
            iaa.WithColorspace(to_colorspace="HSV",
                               from_colorspace="RGB",
                               children=iaa.WithChannels(0, iaa.Add((-200, 200))),
                               name="hue"),
            iaa.WithColorspace(to_colorspace="HSV",
                               from_colorspace="RGB",
                               children=iaa.WithChannels(1, iaa.Add((-20, 20))),
                               name="sat"),
            iaa.ContrastNormalization((0.5, 1.5), per_channel=0.2, name="norm"),
            iaa.Grayscale(alpha=(0.0, 1.0), name="gray"),
        ])),

    # Blur and Noise
    iaa.Sometimes(
        0.2,
        iaa.SomeOf((1, None), [
            iaa.OneOf([iaa.MotionBlur(k=3, name="motion-blur"),
                       iaa.GaussianBlur(sigma=(0.5, 1.0), name="gaus-blur")]),
            iaa.OneOf([
                iaa.AddElementwise((-5, 5), per_channel=0.5, name="add-element"),
                iaa.MultiplyElementwise((0.95, 1.05), per_channel=0.5, name="mul-element"),
                iaa.AdditiveGaussianNoise(scale=0.01 * 255, per_channel=0.5, name="guas-noise"),
                iaa.AdditiveLaplaceNoise(scale=(0, 0.01 * 255), per_channel=True, name="lap-noise"),
                iaa.Sometimes(1.0, iaa.Dropout(p=(0.003, 0.01), per_channel=0.5, name="dropout")),
            ]),
        ],
                   random_order=True)),

    # Colored Blocks
    iaa.Sometimes(0.2, iaa.CoarseDropout(0.02, size_px=(4, 16), per_channel=0.5, name="cdropout")),
])

augs_test = iaa.Sequential([
    iaa.Resize({
        "height": 256,
        "width": 256
    }, interpolation='nearest'),
])
'''


def get_data_loaders():
    train_dataset = SurfaceNormalsDataset(input_dir="../chupao/jpeg2.pickle",
                                          label_dir="../chupao/normal2.pickle",
                                          transform=None,
                                          input_only=None)

    val_dataset = SurfaceNormalsDataset(input_dir='../chupao/jpeg2.pickle',
                                        label_dir='../chupao/normal2.pickle',
                                        transform=None,
                                        input_only=None,
                                        is_train=False)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=64,
                                shuffle=True,
                                num_workers=1,
                                drop_last=False)
    return train_dataloader, val_dataloader


def train(model, train_dataloader, criterion, optimizer, total_iter_num, epoch):
    model.train()
    running_loss = 0.0
    running_mean = 0
    running_median = 0
    for iter_num, batch in enumerate(train_dataloader):
        total_iter_num += 1
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        normal_vectors = model(inputs)
        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)

        loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
        loss /= 64  # Dividing by batch size for drn

        # loss = criterion(normal_vectors, labels)

        loss.backward()
        optimizer.step()

        # normal_vectors_norm = normal_vectors_norm.detach().cpu()
        # normal_vectors = normal_vectors.detach().cpu()
        # labels = labels.detach().cpu()

        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
            normal_vectors_norm, labels.double())
        running_mean += loss_deg_mean.item()
        running_median += loss_deg_median.item()

        # statistics
        running_loss += loss.item()

        writer.add_scalar('data/Train BatchWise Loss', loss.item(), total_iter_num)
        writer.add_scalar('data/Train Mean Error (deg)', loss_deg_mean.item(), total_iter_num)
        writer.add_scalar('data/Train Median Error (deg)', loss_deg_median.item(), total_iter_num)

    # Log Epoch Loss
    num_samples = (len(train_dataloader))
    epoch_loss = running_loss / num_samples
    writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
    print('Train Epoch Loss: {:.4f}'.format(epoch_loss))
    epoch_mean = running_mean / num_samples
    epoch_median = running_median / num_samples
    print('Train Epoch Mean Error (deg): {:.4f}'.format(epoch_mean))
    print('Train Epoch Median Error (deg): {:.4f}'.format(epoch_median))

    if epoch % 100 == 0:
        grid_image = create_grid_image(inputs.detach().cpu(),
                                       normal_vectors_norm.detach().cpu(),
                                       labels.detach().cpu(),
                                       max_num_images_to_save=5)
        writer.add_image('img/training predictions', grid_image, epoch)

    return total_iter_num


def evaluate(model, val_dataloader, criterion, total_iter_num, epoch):
    model.eval()
    running_loss = 0.0
    running_mean = 0
    running_median = 0
    best_loss_deg_mean = 500
    for iter_num, sample_batched in enumerate(val_dataloader):
        inputs, labels = sample_batched
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            normal_vectors = model(inputs)

        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
        loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')

        loss /= 64  # Dividing by batch size for drn
        # loss = criterion(normal_vectors, labels)

        running_loss += loss.item()

        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
            normal_vectors_norm, labels.double())

        if loss_deg_mean.item() < best_loss_deg_mean:
            best_loss_deg_mean = loss_deg_mean.item()
            torch.save(model.state_dict(), 'trained_model_drn_224_size.pth')

        running_mean += loss_deg_mean.item()
        running_median += loss_deg_median.item()

    num_samples = (len(val_dataloader))
    epoch_loss = running_loss / num_samples
    writer.add_scalar('data/Validation Epoch Loss', epoch_loss, total_iter_num)
    print('Validation Epoch Loss: {:.4f}'.format(epoch_loss))
    epoch_mean = running_mean / num_samples
    epoch_median = running_median / num_samples
    print('Val Epoch Mean: {:.4f}'.format(epoch_mean))
    print('Val Epoch Median: {:.4f}'.format(epoch_median))
    writer.add_scalar('data/Val Epoch Mean Error (deg)', epoch_mean, total_iter_num)
    writer.add_scalar('data/Val Epoch Median Error (deg)', epoch_median, total_iter_num)

    if epoch % 100 == 0:
        grid_image = create_grid_image(inputs.detach().cpu(),
                                       normal_vectors_norm.detach().cpu(),
                                       labels.detach().cpu(),
                                       max_num_images_to_save=5)
        writer.add_image('img/validation predictions', grid_image, epoch)


def test(model):
    model.load_state_dict(torch.load('trained_model_drn.pth'))
    model = model.to(device)
    train_dataloader, val_dataloader = get_data_loaders()
    i = 0

    for iter_num, sample_batched in enumerate(val_dataloader):
        if i == 10:
            break
        inputs, labels = sample_batched
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            normal_vectors = model(inputs)

        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)

        grid_image = create_grid_image(inputs.detach().cpu(),
                                       normal_vectors_norm.detach().cpu().float(),
                                       labels.detach().cpu(),
                                       max_num_images_to_save=5)
        writer.add_image('img/validation predictions', grid_image, i)
        i += 1


if __name__ == '__main__':
    model = SurfaceNormalModel(num_classes=3, backbone='drn', freeze_bn=False)
    test(model)

    '''model = SurfaceNormalModel(num_classes=3, backbone='drn', freeze_bn=False)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(0.001),
                                 weight_decay=float(0))

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=float(1e-6),
                                momentum=float(0.9),
                                weight_decay=float(5e-4))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=7,
                                                   gamma=float(0.1))

    criterion = loss_functions.loss_fn_cosine
    # criterion = nn.MSELoss()

    train_dataloader, val_dataloader = get_data_loaders()

    total_iter_num = 0
    epochs = 1000
    for epoch in range(epochs):
        print('\n\nEpoch {}/{}'.format(epoch, epochs - 1))
        writer.add_scalar('data/Epoch Number', epoch, total_iter_num)
        lr_scheduler.step()
        total_iter_num = train(model, train_dataloader, criterion, optimizer, total_iter_num, epoch)
        evaluate(model, val_dataloader, criterion, total_iter_num, epoch)'''
