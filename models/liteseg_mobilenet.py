#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:20:32 2018

@author: Taha Emara  @email: taha@emaraic.com
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from models.backbone_networks import MobileNetV2
from models import aspp
from models.separableconv import SeparableConv2d 




class RT(nn.Module):
    
    def __init__(self, n_classes=19,PRETRAINED_WEIGHTS=".", pretrained=False):
        
        super(RT, self).__init__()
        print("LiteSeg-MobileNet...")

        self.mobile_features=MobileNetV2.MobileNetV2()
        state_dict = torch.load(PRETRAINED_WEIGHTS)
        self.mobile_features.load_state_dict(state_dict)
        

        rates = [1, 3, 6, 9]


        self.aspp1 = aspp.ASPP(1280, 96, rate=rates[0])
        self.aspp2 = aspp.ASPP(1280, 96, rate=rates[1])
        self.aspp3 = aspp.ASPP(1280, 96, rate=rates[2])
        self.aspp4 = aspp.ASPP(1280, 96, rate=rates[3])

        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1280, 96, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(96),
                                             nn.ReLU())
        #self.conv1 = nn.Conv2d(480+1280, 96, 1, bias=False)
        self.conv1 =SeparableConv2d(480+1280,96,1)
        self.bn1 = nn.BatchNorm2d(96)

        # adopt [1x1, 48] for channel reduction.
#            self.conv2 = nn.Conv2d(24, 32, 1, bias=False)
#            self.bn2 = nn.BatchNorm2d(32)
#    
        self.last_conv = nn.Sequential(#nn.Conv2d(24+96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(24+96,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(),
                                       #nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(96,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(),
                                       nn.Conv2d(96, n_classes, kernel_size=1, stride=1))
        
    def forward(self, input):
        x, low_level_features = self.mobile_features(input)
        #print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x,x1, x2, x3, x4, x5), dim=1)
        #print('after aspp cat',x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)
       # ablation=torch.max(low_level_features, 1)[1]
        #print('after con on aspp output',x.size())

#        ##comment to remove low feature
#        low_level_features = self.conv2(low_level_features)
#        low_level_features = self.bn2(low_level_features)
#        low_level_features = self.relu(low_level_features)
        #print("low",low_level_features.size())
        x = torch.cat((x, low_level_features), dim=1)
        #print('after cat low feature with output of aspp',x.size())
        ##   
        
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x#,ablation

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
#if __name__ == "__main__":
#    from flop_count import model_summary 
#    import sys
#    import time
#    from flops_counter import add_flops_counting_methods ,flops_to_string, get_model_parameters_number
#    from torchsummary import summary
#
#    print(sys.path)
#    model = RT(nInputChannels=3, n_classes=19, os=8, pretrained=True, _print=True)
#   
#    model.cuda()
#    model.eval()
#    #input_size = (3, 512, 1024)
#
#    #model_summary.model_summary(model, input_size, query_granularity=1)
#    #summary(model, (3, 360, 640))
#    #image = torch.randn(1, 3, 360, 640).cuda()
##    with torch.no_grad():
##        a = time.perf_counter()
##        output,k = model.forward(image)
##        torch.cuda.synchronize() # wait for mm to finish
##        b = time.perf_counter()
##        print(b-a)
##    print(output.size())
#    
#    total=0
#    for x in range(0,4):
#        print(str(x))
#    for x in range(0,500):
#        image = torch.randn(1, 3, 360, 640).cuda()
#        with torch.no_grad():
#            a = time.perf_counter()
#            output = model.forward(image)
#            torch.cuda.synchronize() # wait for mm to finish
#            b = time.perf_counter()
#            total+=b-a
#    print(str(total/500))
#    total=0
#    for x in range(0,4):
#        print(str(x))
#    for x in range(0,500):
#        image = torch.randn(1, 3, 360, 640).cuda()
#        with torch.no_grad():
#            a = time.perf_counter()
#            output = model.forward(image)
#            torch.cuda.synchronize() # wait for mm to finish
#            b = time.perf_counter()
#            total+=b-a
#    print(str(total/500))
#    batch = torch.FloatTensor(1, 3, 512, 1024).cuda()
#    model = add_flops_counting_methods(model)
#    model.eval().start_flops_count()
#    _ = model(batch)
#    
#    print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
#    print('Params: ' + get_model_parameters_number(model))
#               
##net = MobileNetV2(n_class=1000).cuda()
##state_dict = torch.load('mobilenet_v2.pth') # add map_location='cpu' if no gpu
##net.load_state_dict(state_dict)