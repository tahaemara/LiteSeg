import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict
import numpy as np
import yaml
from addict import Dict
import argparse

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import cityscapes
from dataloaders import utils
from dataloaders import augmentation as augment
from models.liteseg import LiteSeg
from utils import loss as losses
from utils import iou_eval


#To make reproducible results  
torch.manual_seed(125)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(125)
CONFIG=Dict(yaml.load(open("config/training.yaml")))


ap = argparse.ArgumentParser()
ap.add_argument('--backbone_network', required=True,
                help = 'name of backbone network',default='darknet')#shufflenet, mobilenet, and darknet
ap.add_argument('--model_path_coarse', required=False,
                help = 'path to pretrained model on coarse data',default='pretrained_models/liteseg-darknet-cityscapes.pth')
ap.add_argument('--model_path_resume', required=False,
                help = 'path to a model to resume from',default='pretrained_models/liteseg-darknet-cityscapes.pth')

args = ap.parse_args()
backbone_network=args.backbone_network
model_path_resume=args.model_path_resume
model_path_coarse=args.model_path_coarse




# Setting parameters
nEpochs =100  # Number of epochs for training 150
resume_epoch = 0  # Default is 0, change if want to resume 0

p = OrderedDict()  # Parameters to include in report
p['trainBatch'] =4  # Training batch size
p['lr'] =1e-7# Learning rate  1e-8 for darknet and 1e-7 shufflenet and mobilenet
p['wd'] = 5e-4  # Weight decay
p['momentum'] = 0.9  # Momentum
p['epoch_size'] =5  # epochs to change learning rate

testBatch = 1  # Testing batch size
nValInterval = 2  # Run on test set every nTestInterval epochs
snapshot = 2  # Store a model every snapshot epochs


save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

dataset_path=CONFIG.DATASET_FINE
if CONFIG.USING_COARSE:
    print("Taining on Coarse Data")
    dataset_path=CONFIG.DATASET_COARSE
    p['epoch_size'] =10 #we increase the number of epochs to change LR as we train on one scale



exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

class_weight = np.array([0.05570516, 0.32337477, 0.08998544, 1.03602707, 1.03413147, 1.68195437,
                                 5.58540548, 3.56563995, 0.12704978, 1.,         0.46783719, 1.34551528,
                                 5.29974114, 0.28342531, 0.9396095,  0.81551811, 0.42679146, 3.6399074,
                                 2.78376194], dtype=float)
class_weight = torch.from_numpy(class_weight).float().cuda()



#make a folder -with name of current time- for every experiment
experiment_id=datetime.now().strftime("%Y-%m-%d_%H_%M")
save_path = os.path.join(save_dir_root, 'experiments', 'experiment_' + str(experiment_id))
print(save_path)


# Network definition
net=LiteSeg.build(backbone_network,None,CONFIG)
if CONFIG.USING_GPU:
    torch.cuda.set_device(device=CONFIG.GPU_ID)
    net.cuda()


#using the trained model on the coarse data
#If you want to train model on fine data directley, comment the next 3 lines.
if not CONFIG.USING_COARSE:
    print("Using a weights from training coarse data from: {}...".format(model_path_coarse))
    net.load_state_dict(torch.load(model_path_coarse)) 


#resume tarining from a given model, 
#Attention! the learnig rate which used for resuming training, is not the intial one.
if resume_epoch == 0:
    print("Training Network...")
else:
    print("Resume training from a model at: {}...".format(model_path_resume))
    net.load_state_dict(torch.load(model_path_resume))

    
modelName = 'LiteSeg-' + backbone_network + '-cityscapes'
print(modelName)

criterion = losses.cross_entropy2d


if resume_epoch != nEpochs+1:
    # Logging into Tensorboard
    log_dir = os.path.join(save_path, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    #optimizer = optim.Adam(net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=1e-4) 
    p['optimizer'] = str(optimizer)

    composed_transforms_tr = transforms.Compose([
        augment.RandomHorizontalFlip(),
        augment.RandomScale((0.2, .8)),
        augment.RandomCrop(( 512,1024)),
        augment.RandomRotate(5),
        augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        augment.ToTensor()])
   
    composed_transforms_tr1 = transforms.Compose([
        augment.RandomHorizontalFlip(),
        augment.RandomScale((0.2, .8)),
        augment.RandomCrop(( 768,1536)),
        augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        augment.ToTensor()])
    composed_transforms_tr2 = transforms.Compose([
        augment.RandomHorizontalFlip(),
        augment.RandomScale((0.2, .8)),
        augment.RandomCrop(( 360,640)),
        augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        augment.ToTensor()])
    composed_transforms_tr3 = transforms.Compose([
        augment.RandomHorizontalFlip(),
        augment.RandomScale((0.2, .8)),
        augment.RandomCrop(( 720,1280)),
        augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        augment.ToTensor()])
    
    composed_transforms_ts = transforms.Compose([
        augment.RandomHorizontalFlip(),
        #augment.Scale((819, 1638)),
        augment.CenterCrop(( 512,1024)),
        augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),#augment. Normalize_cityscapes(mean=(72.39, 82.91, 73.16)),
        augment.ToTensor()])

    cityscapes_train = cityscapes.Cityscapes(root=dataset_path,extra=CONFIG.USING_COARSE,split='train',transform=composed_transforms_tr)
    cityscapes_train1 = cityscapes.Cityscapes(root=dataset_path,extra=CONFIG.USING_COARSE,split='train',transform=composed_transforms_tr1)
    cityscapes_train2 = cityscapes.Cityscapes(root=dataset_path,extra=CONFIG.USING_COARSE,split='train',transform=composed_transforms_tr2)
    cityscapes_train3 = cityscapes.Cityscapes(root=dataset_path,extra=CONFIG.USING_COARSE,split='train',transform=composed_transforms_tr3)
    
    cityscapes_val = cityscapes.Cityscapes(root=dataset_path,extra=CONFIG.USING_COARSE,split='val', transform=composed_transforms_ts)

    trainloader = DataLoader(cityscapes_train, batch_size=p['trainBatch'], shuffle=True, num_workers=0)
    trainloader1 = DataLoader(cityscapes_train1, batch_size=p['trainBatch'], shuffle=True, num_workers=0)
    trainloader2 = DataLoader(cityscapes_train2, batch_size=p['trainBatch'], shuffle=True, num_workers=0)
    trainloader3 = DataLoader(cityscapes_train3, batch_size=p['trainBatch'], shuffle=True, num_workers=0)


    valloader = DataLoader(cityscapes_val, batch_size=testBatch, shuffle=True, num_workers=0)
    
    if CONFIG.USING_COARSE:#in case of training coarse data, I just used one scale to train.
        loaders=[ trainloader ]
    else:
        loaders=[ trainloader ,trainloader1 ,trainloader2 ,trainloader3]
    
    utils.generate_param_report(os.path.join(save_path, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    num_img_vl = len(valloader)
    running_loss_tr = 0.0
    running_loss_vl = 0.0
    previous_miou = -1.0
    global_step = 0
    iev = iou_eval.Eval(20,19)
    
    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        if epoch % p['epoch_size'] == p['epoch_size'] - 1:
            lr_ = utils.lr_poly(p['lr'], epoch, nEpochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])
           
        net.train()
        for loader in loaders:
            print(loader)
            for ii, sample_batched in enumerate(loader):
                
    
                inputs, labels = sample_batched['image'], sample_batched['label']
                # Forward-Backward of the mini-batch
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                #print('labels size', inputs.size() , labels.size())
                global_step += inputs.data.shape[0]
                #print("Glopal Step",global_step)4,8,12,16
                if CONFIG.USING_GPU:
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                optimizer.zero_grad()
                outputs = net.forward(inputs)
                loss = criterion(outputs, labels,reduct='sum',weight=None)#sum
                loss.backward()
                optimizer.step()
                ls=loss.item()
                running_loss_tr += ls
    #            if ii% 10 == 0:
    #                print(ls)
                # Print stuff
                if ii % num_img_tr == (num_img_tr - 1):
                    running_loss_tr = running_loss_tr / num_img_tr
                    writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                    print('Loss: %f' % running_loss_tr)
                    running_loss_tr = 0
                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time) + "\n")
       
                
                # Update the weights once in p['nAveGrad'] forward passes 
                writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                
    
                # Show 10 * 3 images results each epoch
                if ii % (num_img_tr // 10) == 0:
                    grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                    writer.add_image('Image', grid_image, global_step)
                    grid_image = make_grid(
                        utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy(), 'cityscapes'), 3,
                        normalize=False,
                        range=(0, 255))
                    writer.add_image('Predicted label', grid_image, global_step)
                    grid_image = make_grid(
                        utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy(), 'cityscapes'), 3,
                        normalize=False, range=(0, 255))
                    writer.add_image('Groundtruth label', grid_image, global_step)

        # One testing epoch
        if (epoch % nValInterval == (nValInterval - 1)) or epoch==0:
            total_miou = 0.0
            net.eval()
            for ii, sample_batched in enumerate(valloader):
                inputs, labels = sample_batched['image'], sample_batched['label']

                # Forward pass of the mini-batch
                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                if CONFIG.USING_GPU:
                    inputs, labels = inputs.cuda(), labels.cuda()

                with torch.no_grad():
                    outputs = net.forward(inputs)

                predictions = torch.max(outputs, 1)[1]

                loss = criterion(outputs, labels,reduct='sum',weight=None)#sum elementwise_mean
                running_loss_vl += loss.item()
               
                
                y = torch.ones(labels.size()[2], labels.size()[3]).mul(19).cuda()
                labels=labels.where(labels !=255, y)
                
                iev.addBatch(predictions.unsqueeze(1).data,labels)
                
                
                # Print stuff
                if ii % num_img_vl == num_img_vl - 1:
                    miou=iev.getIoU()[0]
                    running_loss_vl = running_loss_vl / num_img_vl
                    print('Validation:')
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * testBatch + inputs.data.shape[0]))
                    writer.add_scalar('data/test_loss_epoch', running_loss_vl, epoch)
                    writer.add_scalar('data/test_miour', iev.getIoU()[0], epoch)
                    print('Loss: %f' % running_loss_vl)
                    print("Predi iou",iev.getIoU())
                    running_loss_vl = 0
                    iev.reset()

        # Save the model
        if (epoch % snapshot) == snapshot - 1 :#and miou > previous_miou
            previous_miou = miou
            torch.save(net.state_dict(), os.path.join(save_path, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(
                os.path.join(save_path, 'models', modelName + '_epoch-' + str(epoch) + '.pth')))

    writer.close()
