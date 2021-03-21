#!/usr/bin/env python
# Sample script to recreate week1 experiments with minimal code uisng the fastai library

from fastai.vision.all import *
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch
from pathlib import Path

def get_dls(path, bs=64):
    datablock = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                  get_items=get_image_files, 
                  splitter=GrandparentSplitter(train_name='train',valid_name='test'),
                  #item_tfms=Resize(460),
                  batch_tfms=[*aug_transforms(size=64, min_scale=0.75),
                              Normalize.from_stats([0.4273, 0.4523, 0.4497],[0.2567, 0.2470, 0.2766])],
                  get_y=parent_label)
    return datablock.dataloaders(path,bs=bs)


def block(ni, nf):
    return ConvLayer(ni, nf, stride=2)

def get_model():
    return nn.Sequential(
        block(3, 16),
        block(16, 32),
        block(32, 64),
        block(64, 128),
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(128, 8))

if __name__ == '__main__':

    path = Path('/home/group02/mcv/datasets/MIT_split')
    dls = get_dls(path)

    learn = Learner(dls, get_model(), loss_func=F.cross_entropy, opt_func=Adam, metrics=accuracy,cbs = [CudaCallback])
    # use inbuilt learning rate finder
    lr_min, lr_steep = learn.lr_find()

    # use OneCyclePolicy for Training
    learn.fit_one_cycle(15 ,lr_max=lr_steep)
