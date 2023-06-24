import argparse
import json
import os
import random
import shutil
import time
import warnings
import yaml
import torch

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from math import cos, pi, ceil
from src.mobilenetv3 import mobilenetv3
from src.eval_classifier import validate
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import argparse
import os
import yaml

from torchvision.transforms import transforms
from torchvision.datasets.stl10 import STL10


def get_model(num_classes=10, pretrained=True, device=torch.device("cpu")):
    # create model
    model = mobilenetv3.mobilenetv3_large(num_classes=num_classes, width_mult=1.0)
    if pretrained:
        state_dict = torch.load('src/mobilenetv3/pretrained/mobilenetv3-large-1cd25616.pth')
        state_dict.pop("classifier.3.weight")
        state_dict.pop("classifier.3.bias")
        model.load_state_dict(state_dict, strict=False)

    model = torch.nn.DataParallel(model).to(device)
    return model


def resume_model(model, checkpoint_path, optimizer=None, best=False):
    best_prec1 = 0.0
    if not os.path.isdir(checkpoint_path):
        mkdir_p(checkpoint_path)

    ckpt = "checkpoint.pth.tar" if not best else "model_best.pth.tar"
    checkpoint_file = os.path.join(checkpoint_path, ckpt)
    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint {checkpoint_file} (epoch {epoch})")


    else:
        epoch = 0
        print(f"=> no checkpoint found at {checkpoint_file}")

    return model, epoch, best_prec1


def get_transforms(split='train', input_size=(96, 96)):
    if split == 'train':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.RandomResizedCrop(input_size, antialias=True),
            transforms.RandomHorizontalFlip(),
        ])
    elif split == 'test':
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()
        ])
    return transform


def get_dataset(batch_size, workers=4):
    input_shape = (96, 96)
    num_classes = 10
    train_dataset = STL10(root="./data/stl10", download=True, split="train",
                          transform=get_transforms(split='train', input_size=input_shape), )
    val_dataset = STL10(root="./data/stl10", download=True, split="test",
                        transform=get_transforms(split='test', input_size=input_shape))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    train_loader_len = ceil(len(train_dataset) / batch_size)
    val_loader_len = ceil(len(val_dataset) / batch_size)

    return train_loader, val_loader, train_loader_len, val_loader_len


class LRAdjust:
    def __init__(self, lr, warmup, epochs):
        self.lr = lr
        self.warmup = warmup
        self.epochs = epochs

    def adjust(self, optimizer, epoch, iteration, num_iter):
        gamma = 0.1
        warmup_epoch = 5 if self.warmup else 0
        warmup_iter = warmup_epoch * num_iter
        current_iter = iteration + epoch * num_iter
        max_iter = self.epochs * num_iter
        lr = self.lr * (gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))

        if epoch < warmup_epoch:
            lr = self.lr * current_iter / warmup_iter

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
