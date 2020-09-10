import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ggrasp.ggrasp import post_process_output
from ggrasp.losses import smoothness_loss
from ggrasp.utils.utils import import_string
from ggrasp.utils.dataset_processing import evaluation

parser = argparse.ArgumentParser(description='Train Generative Grasping Models.')

# Network
parser.add_argument('--network', type=str, default='GRConvNet',
                    help='Network name in inference/models')
parser.add_argument('--use-depth', type=int, default=1,
                    help='Use Depth image for training (1/0)')
parser.add_argument('--use-rgb', type=int, default=1,
                    help='Use RGB image for training (1/0)')
parser.add_argument('--channel-size', type=int, default=32,
                    help='Internal channel size for the network')

# Datasets
parser.add_argument('--dataset', type=str,
                    help='Dataset class name. (e.g. CornellDataset)')
parser.add_argument('--dataset-path', type=str,
                    help='Path to dataset',
                    required=True)

# Training
parser.add_argument('--batch-size', type=int, default=8,
                    help='Batch size')
parser.add_argument('--epochs', type=int, default=50,
                    help='Training epochs')
parser.add_argument('--lr', default=0.001,
                    type=float,
                    help='Learning rate.')
parser.add_argument("--smoothness_weight", type=float,
                    default=0.5,
                    help="Weightage for Smoothness loss")
parser.add_argument('--loss', type=str,
                    help="The loss function to use.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()

args = parser.parse_args()

tb = SummaryWriter(".")

if args is not None:
    with open('args.json', 'w') as f:
        json.dump(vars(args), f)

device = torch.device("cuda:0")

# Load Dataset
logging.info('Loading {}...'.format(args.dataset))
Dataset = import_string("ggrasp.datasets.{}".format(args.dataset))
dataset = Dataset(args.dataset_path,
                  random_rotate=True,
                  random_zoom=True,
                  include_depth=args.use_depth,
                  include_rgb=args.use_rgb)
logging.info('Dataset size is {}'.format(dataset.length))

# Creating data indices for training and validation splits
dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

log.info('Training size: {}'.format(train_size))
log.info('Validation size: {}'.format(val_size))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=8,
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,               # Must be 1
    num_workers=8,
    shuffle=True
)

# Load the network
log.info('Loading Network...')
input_channels = 1 * args.use_depth + 3 * args.use_rgb
network = import_string("ggrasp.models.{}".format(args.network))
net = network(input_channels=input_channels)
net = net.to(device)
loss_fn = import_string("ggrasp.losses.{}".format(args.loss))

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

best_iou = 0.0

def validate(net, device, val_data):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {}
    }

    ld = len(val_data)

    with torch.no_grad():
        for x, y, didx, rot, zoom_factor in val_data:
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            y_pos, y_cos, y_sin, y_width = yc
            pos_pred, cos_pred, sin_pred, width_pred = net(xc)

            p_loss = loss_fn(pos_pred, y_pos)
            cos_loss = loss_fn(cos_pred, y_cos)
            sin_loss = loss_fn(sin_pred, y_sin)
            width_loss = loss_fn(width_pred, y_width)

            p_smooth_loss = smoothness_loss(pos_pred)
            cos_smooth_loss = smoothness_loss(cos_pred)
            sin_smooth_loss = smoothness_loss(sin_pred)
            width_smooth_loss = smoothness_loss(width_pred)

            lossd = {
                'loss': p_loss + cos_loss + sin_loss + width_loss + args.smoothness_weight * (p_smooth_loss + cos_smooth_loss + sin_smooth_loss + width_smooth_loss),
                'losses': {
                    'p_loss': p_loss,
                    'cos_loss': cos_loss,
                    'sin_loss': sin_loss,
                    'width_loss': width_loss,
                    'p_smooth_loss': p_smooth_loss,
                    'cos_smooth_loss': cos_smooth_loss,
                    'sin_smooth_loss': sin_smooth_loss,
                    'width_smooth_loss': width_smooth_loss
                },
                'pred': {
                    'pos': pos_pred,
                    'cos': cos_pred,
                    'sin': sin_pred,
                    'width': width_pred
                }
            }

            loss = lossd['loss']

            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            s = evaluation.calculate_iou_match(q_out,
                                               ang_out,
                                               val_data.dataset.dataset.get_gtbb(didx, rot, zoom_factor),
                                               no_grasps=1,
                                               grasp_width=w_out,
                                               )

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    for x, y, _, _, _ in train_data:
        xc = x.to(device)
        yc = [yy.to(device) for yy in y]
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = net(xc)

        p_loss = loss_fn(pos_pred, y_pos)
        cos_loss = loss_fn(cos_pred, y_cos)
        sin_loss = loss_fn(sin_pred, y_sin)
        width_loss = loss_fn(width_pred, y_width)

        p_smooth_loss = smoothness_loss(pos_pred)
        cos_smooth_loss = smoothness_loss(cos_pred)
        sin_smooth_loss = smoothness_loss(sin_pred)
        width_smooth_loss = smoothness_loss(width_pred)

        lossd = {
            'loss': p_loss + cos_loss + sin_loss + width_loss + args.smoothness_weight * (p_smooth_loss + cos_smooth_loss + sin_smooth_loss + width_smooth_loss),
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss,
                'p_smooth_loss': p_smooth_loss,
                'cos_smooth_loss': cos_smooth_loss,
                'sin_smooth_loss': sin_smooth_loss,
                'width_smooth_loss': width_smooth_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

        loss = lossd['loss']

        results['loss'] += loss.item()

        for ln, l in lossd['losses'].items():
            if ln not in results['losses']:
                results['losses'][ln] = 0
            results['losses'][ln] += l.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return results

for epoch in range(args.epochs):
    log.info('Beginning Epoch {:02d}'.format(epoch))
    train_results = train(epoch, net, device, train_loader, optimizer)

    # Log training losses to tensorboard
    tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
    for n, l in train_results['losses'].items():
        tb.add_scalar('train_loss/' + n, l, epoch)

    # Run Validation
    log.info('Validating...')
    test_results = validate(net, device, val_loader)
    log.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                             test_results['correct'] / (test_results['correct'] + test_results['failed'])))

    # Log validation results to tensorboard
    tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
    tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
    for n, l in test_results['losses'].items():
        tb.add_scalar('val_loss/' + n, l, epoch)

    # Save best performing network
    iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
    if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
        torch.save(net, 'epoch_%02d_iou_%0.2f' % (epoch, iou))
        best_iou = iou
