import os
import sys
from PIL.Image import FASTOCTREE
import torch
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import torch.nn as nn
# Util function for loading meshes
from torch.utils.tensorboard import SummaryWriter

# Data structures and functions for rendering
from s2mtest import s2m_test

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

from backup_utils import backup_terminal_outputs, backup_code, set_seed
set_seed(1000)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")



from dataset import EarSytheticDataset
from torch.utils.data import DataLoader
import math
import time
from utils import get_face_mask
from torchvision.utils import save_image
from config import cfg

import FCRN




save_path = os.path.join('./log', time.strftime("%y%m%d_%H%M%S")+'_depth')
os.makedirs(save_path, exist_ok=True)

backup_terminal_outputs(save_path)
backup_code(save_path, marked_in_parent_folder=[])
writer = SummaryWriter(save_path)


batch_size = 16
img_size = 256

use_background = False

train_set = EarSytheticDataset(cfg.model.sythetic_dataset_path, train = True, input_size=img_size, back_ground = use_background)
test_set = EarSytheticDataset(cfg.model.sythetic_dataset_path, train = False, input_size=img_size, back_ground = use_background)


trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


model = FCRN.ResNet(layers=34, output_size=(256, 256)).to(device)

train_params = [{'params': model.get_1x_lr_params(), 'lr': 0.01},
                        {'params': model.get_10x_lr_params(), 'lr': 0.01 * 10}]


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,200], gamma=0.2)
mseloss = nn.MSELoss()




best_loss = 10000

best_e=0

first_epoch = True
begin_epoch = 0


TOT_EPOCH = 100

def depth_loss(gt_depth,pred_depth,mask):
    h,w = gt_depth.shape[1:]

    
    log_gt = torch.log(gt_depth)
    log_pred = torch.log(pred_depth)
    

    alpha = torch.sum((log_gt - log_pred).view(-1,h*w),dim=1) / (h*w)
    # print(log_gt.sum(),h,w)
    loss = torch.sum(((log_pred - log_gt + alpha[:,None,None])**2).view(-1,h*w),dim=1) / (2*h*w)
    loss = torch.mean(loss)
    return loss

def depth_mse_loss(gt_depth,pred_depth,mask):
    h,w = gt_depth.shape[1:]

    
    log_gt = torch.log(gt_depth)
    log_pred = torch.log(pred_depth)
    

  
    loss = torch.sum(((log_pred - log_gt)**2).view(-1,h*w),dim=1) / (2*h*w)
    loss = torch.mean(loss)
    return loss
def depth_mse_loss_no_log(gt_depth,pred_depth,mask):
    h,w = gt_depth.shape[1:]
  
    loss = torch.sum(((pred_depth - gt_depth)**2).view(-1,h*w),dim=1) / (2*h*w)
    loss = torch.mean(loss)
    return loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target,mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (mask>0).detach()
        diff = target - pred
       
        diff = diff[valid_mask]
        
        self.loss = diff.abs().mean()
        return self.loss

log_epoch = np.arange(0,TOT_EPOCH,int(TOT_EPOCH/5))

mask_loss = MaskedL1Loss()

for e in range(begin_epoch):
    scheduler.step()
best_test_loss = 0
for e in range(begin_epoch, TOT_EPOCH):
    print('###################')
    print('Epoch:', e)
    print('###################')
    model.train()
    
    avg_loss=0
    cnt=0

    if e in log_epoch:
        epoch_train_path = os.path.join(save_path,'epoch_%d/train/'%e)
        epoch_test_path = os.path.join(save_path,'epoch_%d/test/'%e)
        
        os.makedirs(epoch_train_path, exist_ok=True)
        os.makedirs(epoch_test_path, exist_ok=True)
     
        latent_f = open(os.path.join(epoch_train_path, 'latent_code.txt'), 'w')
        latent_f_test = open(os.path.join(epoch_test_path, 'latent_code.txt'), 'w')
    start = time.time()
    for i, data in enumerate(trainloader):
        
        bs = data['image'].shape[0]


        gt_img = data['image'].to(device)
       
        gt_masks = data['mask'].to(device)

        gt_depth = data['depth'].to(device).unsqueeze(1)
        masked_gt = (gt_img*gt_masks).to(device)

        out,_ = model(masked_gt)
        losses = {}

        gt_depth[gt_depth<=0] = 1e-5

        pred_depth = out +1e-5 
        losses['depth'] = mask_loss(pred_depth,gt_depth,gt_masks) 
     
        all_loss = 0.
        losses_key = losses.keys()

        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        # depth_loss.backward()
        all_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        avg_loss += all_loss.item()
        cnt += 1

        loss_display = {}
        
        for k,v in losses.items():
            loss_display[k] = v.item()
        if i % 20 == 0:
            print(('Iter: {}|{} ').format(i, len(trainloader)), loss_display)
        # exit()
        
        rand=np.random.rand(1)
        if e in log_epoch and rand[0]<0.02: 

            for k, (img1, img2) in enumerate(zip((gt_depth).detach().cpu(), (pred_depth).detach().cpu())):
                
                
                np.save(epoch_train_path+'/pred_%d.npy'%(i*batch_size+k),img2)
                np.save(epoch_train_path+'/gt_%d.npy'%(i*batch_size+k),img1)
      
           
           
   

    
    print('train_loss: {:.4f} time: {:.4f}'.format(avg_loss/cnt, time.time() - start))

    if avg_loss/cnt<best_loss:
        torch.save(model.state_dict(), os.path.join(save_path, "depth_estimate_train.pt"))
        best_loss = avg_loss/cnt


    scheduler.step()
    model.eval()
    test_cnt = 0
    avg_test_loss = 0

    with torch.no_grad():
        for i,data in enumerate(testloader):
            bs = data['image'].shape[0]


            gt_img = data['image'].to(device)
        
            gt_masks = data['mask'].to(device)

            gt_depth = data['depth'].to(device).unsqueeze(1)
            masked_gt = (gt_img*gt_masks).to(device)
            #print(gt_masks.shape)

            out,_ = model(masked_gt)
            # print(out.shape)
            losses = {}
            # print(out[0].min())
            # pred_depth = out[0] +1e-5
            gt_depth[gt_depth<=0] = 1e-5

            pred_depth = out +1e-5 
            # pred_depth = out[0].squeeze(1).contiguous() +1e-5 
            losses['depth'] = mask_loss(pred_depth,gt_depth,gt_masks)
        
            all_loss = 0.
            losses_key = losses.keys()

            for key in losses_key:
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
                
            test_cnt += 1
            avg_test_loss += all_loss
            loss_display = {}
            for k,v in losses.items():
                loss_display[k] = v.item()
            if i % 20 == 0:
                print(('Iter: {}|{} ').format(i, len(testloader)), loss_display)
        
            # break
        if (e%5==0 and e!=begin_epoch) or first_epoch: 
            torch.save(model.state_dict(), os.path.join(save_path, "depth_estimator_%d.pt"%e))
        rand=np.random.rand(1)
        if e in log_epoch and rand<0.1: 

            for k, (img1, img2) in enumerate(zip((gt_depth).detach().cpu(), (pred_depth).detach().cpu())):
                
                
                np.save(epoch_test_path+'/pred_%d.npy'%(i*batch_size+k),img2)
                np.save(epoch_test_path+'/gt_%d.npy'%(i*batch_size+k),img1)
                
            
    if avg_test_loss/test_cnt < best_test_loss:
        torch.save(encoder.state_dict(), os.path.join(save_path, "resnet_encoder.pt"))
        best_test_loss = avg_test_loss/test_cnt
    print('test_loss:',avg_test_loss/test_cnt)
    first_epoch = False

print('best_test_loss:', best_test_loss)





