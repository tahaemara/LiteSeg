#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:38:30 2018

@author: Taha Emara  @email: taha@emaraic.com
"""

import time
import numpy as np
import argparse
from addict import Dict
import yaml
import cv2


# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms 
from torch.utils.data import DataLoader

# Custom includes
from dataloaders import cityscapes
from dataloaders import utils as dataloaders_utils
from models.liteseg import LiteSeg
from utils import iou_eval
from dataloaders import augmentation as augment
from utils.flops_counter import add_flops_counting_methods ,flops_to_string, get_model_parameters_number

#from models.backbone_networks.darknet import Darknet19

ap = argparse.ArgumentParser()
ap.add_argument('--backbone_network', required=False,
                help = 'name of backbone network',default='shufflenet')#shufflenet, mobilenet, and darknet
ap.add_argument('-modpth', '--model_path', required=False,
                help = 'path to pretrained model',default='pretrained_models/liteseg-shufflenet-cityscapes.pth')


CONFIG=Dict(yaml.load(open("config/training.yaml")))


args = ap.parse_args()
backbone_network=args.backbone_network
modelpath=args.model_path


#Net1=Darknet19(weights_file=CONFIG.PRETRAINED_DarkNET19)
#Net1.cuda()
#Net1.eval()

net=LiteSeg.build(backbone_network,modelpath,CONFIG,is_train=False)
net.eval()  


if CONFIG.USING_GPU:
    torch.cuda.set_device(device=CONFIG.GPU_ID)
    net.cuda()

#burn-in with 200 images   
for x in range(0,200):
    image = torch.randn(1, 3, 360, 640).cuda()
    with torch.no_grad():
        output = net.forward(image)


#reporting results in fps:  
total=0
for x in range(0,200):
    image = torch.randn(1, 3, 360, 640).cuda()
    with torch.no_grad():
        a = time.perf_counter()
        output = net.forward(image)
        torch.cuda.synchronize()
        b = time.perf_counter()
        total+=b-a
print(str(200/total))

batch = torch.FloatTensor(1, 3, 512, 1024).cuda()
model = add_flops_counting_methods(net)
model.eval().start_flops_count()
_ = model(batch)



print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
print('Params: ' + get_model_parameters_number(model))


composed_transforms_ts = transforms.Compose([
                        augment.RandomHorizontalFlip(),                
                        augment.CenterCrop((512,1024)),
                        augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),#Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),#augment. Normalize_cityscapes(mean=(72.39, 82.91, 73.16)),#123.15, 115.90, 103.06 72.39, 82.91, 73.16
                        augment.ToTensor()])
    
   

cityscapes_val = cityscapes.Cityscapes(root=CONFIG.DATASET_FINE,extra=CONFIG.USING_COARSE,split='val', transform=composed_transforms_ts)
valloader = DataLoader(cityscapes_val, batch_size=1, shuffle=True, num_workers=0)


num_img_vl = len(valloader)
iev = iou_eval.Eval(20,19)


net.eval()
for ii, sample_batched in enumerate(valloader):#valloader
            inputs, labels = sample_batched['image'], sample_batched['label']
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            if CONFIG.USING_GPU:
                inputs, labels = inputs.cuda(), labels.cuda()

            with torch.no_grad():
                outputs = net.forward(inputs)
                
            predictions = torch.max(outputs, 1)[1]
            y = torch.ones(labels.size()[2], labels.size()[3]).mul(19).cuda()
            labels=labels.where(labels !=255, y)
            
            iev.addBatch(predictions.unsqueeze(1).data,labels)
            if ii % num_img_vl == num_img_vl - 1 :
                print('Validation Result:')
                print("MIOU",iev.getIoU())