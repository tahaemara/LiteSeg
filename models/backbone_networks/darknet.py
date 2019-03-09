#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:47:01 2018

@author: Taha Emara  @email: taha@emaraic.com 'based on lightnet library by EAVISE' 
"""

from collections import OrderedDict
import torch.nn as nn
import lightnet.network as lnn



class Darknet19(lnn.module.Darknet):
    """ `Darknet19`_ implementation with pytorch.

    Args:
        weights_file (str, optional): Path to the saved weights; Default **None**
        input_channels (Number, optional): Number of input channels; Default **3**

    .. _Darknet19: https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    """

    def __init__(self, weights_file=None, input_channels=3):
        """ Network initialisation """
        super().__init__()


        # Network
        self.layers = nn.Sequential(
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('6_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0)),
                ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('8_max',           nn.MaxPool2d(2,2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),#2>1
                ('10_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                ('11_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('12_max',          nn.MaxPool2d(2,2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1,dilation=1)),
                ('14_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1,1, 1,dilation=1)),
                ('15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1,dilation=1)),
                ('16_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 1,dilation=1)),
                ('17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1,dilation=1)),
                ('18_max',          nn.MaxPool2d(1,1)),
                ('19_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1,dilation=2)),
                ('20_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 1,dilation=2)),
                ('21_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1,dilation=2)),
                ('22_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0,dilation=2)),
                ('23_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1,dilation=2))
            ])
        )

        if weights_file is not None:
            self.load(weights_file)

    def _forward(self, x):
        i=0
        for i,l in enumerate( self.layers[:7]):
            #print(str(i),x.size())
            x = l(x)
        keep = x
        for z,l in enumerate(self.layers[7:]):
            #print(str(z+i),x.size())
            x = l(x)
        return x,keep

""" os =16
   ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('6_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0)),
                ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('8_max',           nn.MaxPool2d(2,2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),#2>1
                ('10_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                ('11_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1,dilation=1)),
                ('14_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1,1, 1,dilation=1)),
                ('15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1,dilation=1)),
                ('16_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 1,dilation=1)),
                ('17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1,dilation=1)),
                ('18_max',          nn.MaxPool2d(1,1)),
                ('19_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1,dilation=2)),
                ('20_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 1,dilation=2)),
                ('21_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1,dilation=2)),
                ('22_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0,dilation=2)),
                ('23_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1,dilation=2))
                
"""
"""os =32
 ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('6_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0)),
                ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('8_max',           nn.MaxPool2d(2,2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),#2>1
                ('10_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                ('11_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1,dilation=2)),
                ('14_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1,1, 0,dilation=2)),
                ('15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1,dilation=2)),
                ('16_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0,dilation=2)),
                ('17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1,dilation=2)),
                ('18_max',          nn.MaxPool2d(1,1)),
                ('19_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1,dilation=4)),
                ('20_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0,dilation=4)),
                ('21_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1,dilation=4)),
                ('22_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0,dilation=4)),
                ('23_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1,dilation=4))
"""
