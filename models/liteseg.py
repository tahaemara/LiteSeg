#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 12:00:52 2019

@author: Taha Emara  @email: taha@emaraic.com
"""
import torch 

from models import liteseg_shufflenet as shufflenet
from models import liteseg_darknet as darknet
from models import liteseg_mobilenet as mobilenet



class LiteSeg():
    
        
    def build(backbone_network,modelpath,CONFIG,is_train=True):
                
        if backbone_network.lower() == 'darknet':
            net = darknet.RT(n_classes=19, pretrained=is_train,PRETRAINED_WEIGHTS=CONFIG.PRETRAINED_DarkNET19)
        elif backbone_network.lower() == 'shufflenet':
            net = shufflenet.RT(n_classes=19, pretrained=is_train, PRETRAINED_WEIGHTS=CONFIG.PRETRAINED_SHUFFLENET)
        elif backbone_network.lower() == 'mobilenet':
            net = mobilenet.RT(n_classes=19,pretrained=is_train, PRETRAINED_WEIGHTS=CONFIG.PRETRAINED_MOBILENET)
        else:
            raise NotImplementedError
            
        if modelpath is not None:
            net.load_state_dict(torch.load(modelpath))
            
        print("Using LiteSeg with",backbone_network)
        return net
        
            
    
