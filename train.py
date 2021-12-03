from attrdict import AttrDict
import argparse
import oyaml
import tqdm
import os

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import torch.nn as nn
import torch

import loss_functions
import dataloader
from model import SurfaceNormalModel

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
args = parser.parse_args()

CONFIG_FILE_PATH = args.configFile

with open(CONFIG_FILE_PATH) as fd:
    config_yaml = oyaml.load(fd)

config = AttrDict(config_yaml)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_dir = "./runs/" + args.title
writer = SummaryWriter(log_dir, comment='create-graph')
checkpoint_dir = log_dir + "/checkpoints"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

input_only = [
    "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
    "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
]

augs_train = iaa.Sequential([
    # Geometric Augs
    iaa.Resize({
        "height": config.train.imgHeight,
        "width": config.train.imgWidth
    }, interpolation='nearest'),
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
        "height": config.train.imgHeight,
        "width": config.train.imgWidth
    }, interpolation='nearest'),
])

def get_data_loaders():
    train_dataset = dataloader.SurfaceNormalsDataset(input_dir=config.train.images,
                                                         label_dir=config.train.labels,
                                                         transform=augs_train,
                                                         input_only=input_only)

    val_dataset = dataloader.SurfaceNormalsDataset(input_dir=config.val.images,
                                                     label_dir=config.val.labels,
                                                     transform=augs_test,
                                                     input_only=None)

    train_dataloader = DataLoader(train_dataset,
                                   batch_size=config.train.batchSize,
                                   shuffle=True,
                                   num_workers=config.train.numWorkers,
                                   drop_last=True,
                                   pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                   batch_size=config.val.validationBatchSize,
                                   shuffle=True,
                                   num_workers=config.val.numWorkers,
                                   drop_last=False)
    return train_dataloader, val_dataloader


def train(model, train_dataloader, criterion, optimizer, total_iter_num):
    model.train()
    running_loss = 0.0
    running_mean = 0
    running_median = 0
    for iter_num, batch in enumerate(tqdm(train_dataloader)):
        total_iter_num += 1
        inputs, labels, masks = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        normal_vectors = model(inputs)
        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)

        loss = criterion(normal_vectors_norm, labels, reduction='sum')

        loss.backward()
        optimizer.step()

        normal_vectors_norm = normal_vectors_norm.detach().cpu()
        inputs = inputs.detach().cpu()
        labels = labels.detach().cpu()
        mask_tensor = masks.squeeze(1)

        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
            normal_vectors_norm, labels.double(), mask_tensor)
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
    return total_iter_num


def evaluate(model, val_dataloader, criterion, total_iter_num):
    model.eval()
    running_loss = 0.0
    running_mean = 0
    running_median = 0
    for iter_num, sample_batched in enumerate(tqdm(val_dataloader)):
        inputs, labels, masks = sample_batched
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            normal_vectors = model(inputs)

        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
        loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')

        running_loss += loss.item()

        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
            normal_vectors_norm, labels.double())
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


model = SurfaceNormalModel(num_classes=config.train.numClasses, backbone='resnet', freeze_bn=False)
#model = nn.DataParallel(model)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=float(config.train.optimAdam.learningRate),
                             weight_decay=float(config.train.optimAdam.weightDecay))

criterion = loss_functions.loss_fn_cosine

train_dataloader, val_dataloader = get_data_loaders()

total_iter_num = 0
epochs = config.train.numEpochs
for epoch in range(epochs):
    print('\n\nEpoch {}/{}'.format(epoch, epochs - 1))
    writer.add_scalar('data/Epoch Number', epoch, total_iter_num)
    total_iter_num = train(model, train_dataloader, criterion, optimizer, total_iter_num)
    evaluate(model, val_dataloader, criterion, total_iter_num)


