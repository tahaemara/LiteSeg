#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 18:57:26 2019

@author: Taha Emara  @email: taha@emaraic.com
"""


import argparse
from addict import Dict
import yaml
import os
from PIL import Image 
import cv2
import glob
# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms 

# Custom includes
from dataloaders import utils as dataloaders_utils
from models.liteseg import LiteSeg


ap = argparse.ArgumentParser()
ap.add_argument('--backbone_network', required=False,
                help = 'name of the archtectirure',default='darknet')#shufflenet'darknet19reduced
ap.add_argument('--model_path', required=False,
                help = 'path to pretrained model',default='pretrained_models/liteseg-darknet-cityscapes.pth')
ap.add_argument('--images_path', required=False,
                help = 'path to pretrained model',default='samples/')

ap.add_argument('--gpu', required=False, dest='gpu',action='store_true',
                help = 'use gpu')
ap.add_argument('--no-gpu', required=False, dest='gpu',action='store_false',
                help = 'use cpu')
ap.set_defaults(gpu=False)

CONFIG=Dict(yaml.load(open("config/training.yaml")))

args = ap.parse_args()
backbone_network=args.backbone_network
modelpath=args.model_path
images_path=args.images_path
use_gpu=args.gpu

net=LiteSeg.build(backbone_network,modelpath,CONFIG)
net.eval()  


if use_gpu:
    torch.cuda.set_device(device=0)
    net.cuda()
    
loader = transforms.Compose([
        transforms.CenterCrop((1024,2048)),
        transforms.ToTensor() ,
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])



images=glob.glob(images_path+'/*.png')

for i, image_path in enumerate(images):
    print("Processing Images ",i)
    img=Image.open(image_path)
    w,h=img.size
    input1=loader(img)
    img2=input1.unsqueeze(0)
    image= Variable(img2, requires_grad=True)
    if use_gpu:
        image= image.cuda()
    oupath=os.path.join("samples","predictions",os.path.basename(image_path)[:-4]+"_liteseg-"+backbone_network+".png")

    with torch.no_grad():
        outputs = net.forward(image)
        predictions = torch.max(outputs, 1)[1]
        off=predictions.detach().cpu().numpy()
    pred_color=dataloaders_utils.decode_segmap_cv(off, 'cityscapes')
    cv2.imwrite(oupath,pred_color)