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

from pytorch3d.io import save_obj
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from models import Render, range_loss, ply_loss, shape_vec_cos_sim, ResNet_FCRN
from dataset import EarExtendDataLoader, EarValImgLoader, EarCombineDataLoader
from torch.utils.data import DataLoader
import math
import time
from utils import get_face_mask, rgb2gray
from torchvision.utils import save_image
from config import cfg


import FCRN



save_path = os.path.join('./log', time.strftime("%y%m%d_%H%M%S")+'_recon_depth_withposeepoch_adjustlr_add')
os.makedirs(save_path, exist_ok=True)
print(save_path)
backup_terminal_outputs(save_path)
backup_code(save_path, marked_in_parent_folder=[])
writer = SummaryWriter(save_path)


batch_size = 8
img_size = 256
load_pretrain = False
with open(cfg.model.pkl_path,'rb') as f:
    ear_model = pickle.load(f)
mu = torch.tensor(ear_model['Mean']).float().to(device)
faces = torch.tensor(ear_model['Trilist']).to(device)
U = torch.tensor(ear_model['Eigenvectors']).float().to(device)
V = torch.tensor(ear_model['EigenValues']).float().to(device)

WS = pd.read_excel(cfg.model.land_mark_path)
WS_np = np.array(WS)
land_mark = torch.tensor(WS_np[WS_np[:,1]>0][:,1]).long().to(device)

# train_set = EarCombineDataLoader(cfg.model.ear_dataset_path, train = True, input_size=img_size,dataset = 'both')
# test_set = EarCombineDataLoader(cfg.model.ear_dataset_path, train = False, input_size=img_size,dataset = 'both')
train_set = EarExtendDataLoader(cfg.model.ear_dataset_path, train = True, input_size=img_size)
test_set = EarExtendDataLoader(cfg.model.ear_dataset_path, train = False, input_size=img_size)



s2mimg_set = EarValImgLoader(cfg.s2m.s2m_data_path, input_size=img_size)
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
s2mimg_loader = DataLoader(s2mimg_set, batch_size=batch_size, shuffle=False, num_workers=4)

fcrn_model = FCRN.ResNet(layers=34, output_size=(256, 256)).to(device)
fcrn_model.load_state_dict(torch.load(cfg.model.depth_model_path))
for param in fcrn_model.parameters():
    param.requires_grad = False
fcrn_model.eval()
encoder = ResNet_FCRN().to(device)

train_parameters = encoder.parameters()
lr=0.001
decoder = Render(mu, V, U, faces, cfg.model).to(device)
optimizer = torch.optim.Adam(train_parameters, lr=0.001)
# optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 90], gamma=0.2)
mseloss = nn.MSELoss()


rot_loss1 = range_loss(-math.pi/3, math.pi/3) # 绕从屏幕穿出的轴
rot_loss2 = range_loss(-1.5, -0.5) # 绕纵轴
rot_loss3 = range_loss(-math.pi/3, math.pi/3) # 绕横轴
f_loss = range_loss(0.5, 4) # scale
light_loss = range_loss(0, 80)

def latent_loss(out):
    return  f_loss(out[:,5]) # + rot_loss2(out[:,1])

cos_sim_costraint = range_loss(-1, 0.8)

best_loss = 10000
best_ldm = 100000
best_loss_list = 0
best_test_loss = 100000
best_test_ldm = 100000
best_test_list = 0
best_e=0


first_epoch = False
begin_epoch = 0


POS_EPOCH = 20 # cfg.train.pos_epoch
SHAPE_EPOCH = cfg.train.pos_epoch
TEXTURE_EPOCH = cfg.train.pos_epoch
TOT_EPOCH = cfg.train.total_epoch

for e in range(begin_epoch):
    scheduler.step()
# s2m_epoch = np.concatenate([np.arange(TOT_EPOCH-1-10,TOT_EPOCH-1,3)],axis=0)
s2m_epoch = [TOT_EPOCH-3]
# log_epoch = np.concatenate([np.arange(POS_EPOCH-5,TOT_EPOCH,int(TOT_EPOCH/5)),[TOT_EPOCH-1]],axis=0)
log_epoch = [POS_EPOCH, 50, TOT_EPOCH-3,]
print('s2m_epoch,log_epoch: ', s2m_epoch,log_epoch)
for e in range(begin_epoch, TOT_EPOCH):
    print('###################')
    print('Epoch:', e)
    print('###################')
    encoder.train()
    
    avg_loss=0
    cnt=0

    if e in s2m_epoch or e in log_epoch:
        epoch_train_path = os.path.join(save_path,'epoch_%d/train/'%e)
        epoch_test_path = os.path.join(save_path,'epoch_%d/test/'%e)
        epoch_s2mimg_path = os.path.join(save_path,'epoch_%d/s2mimg/'%e)
        os.makedirs(epoch_train_path, exist_ok=True)
        os.makedirs(epoch_test_path, exist_ok=True)
        os.makedirs(epoch_s2mimg_path, exist_ok=True)    
        latent_f = open(os.path.join(epoch_train_path, 'latent_code.txt'), 'w')
        latent_f_test = open(os.path.join(epoch_test_path, 'latent_code.txt'), 'w')
    start = time.time()
    for i, data in enumerate(trainloader):
        bs = data[0].shape[0]

        gt_land = data[3].to(device)
        gt_img = data[0].to(device)
        masked_gt = (data[0]*data[2]).to(device)
        gt_masks = data[2].to(device)
        # print(gt_img.shape)
        with torch.no_grad():
            depth, frcn_feat = fcrn_model(masked_gt)
        out, tex, shape_vec = encoder(masked_gt,frcn_feat)
        

        if e<POS_EPOCH:
            
            shape_vec[:,:] = 0
            tex[:,:] = 0

        render_images, project_verts, meshes, tex_maps,_,_ = decoder(out, tex, shape_vec)

        render_mask = render_images[..., 3].clone()
        render_images = render_images[..., :3] * render_images[..., 3:4]
        render_images = render_images.permute(0,3,1,2) # reshape to B, C, H, W

        render_lmks = project_verts[:,land_mark,:2]
        
        render_lmks_c = torch.clamp(render_lmks,0,img_size-1)
        pred_masks = get_face_mask(img_size, render_lmks_c.detach().cpu().long().numpy())
        masked_render = render_images * pred_masks.to(device)

        lmk_bbox_length = torch.sqrt(torch.sum((torch.max(gt_land, dim=1).values - torch.min(gt_land, dim=1).values)**2,dim=1))
        land_loss = 10 * torch.mean(torch.mean(torch.sqrt(torch.sum((gt_land - render_lmks)**2,dim=2)), dim=1) / lmk_bbox_length)

        contour_loss = 100 * ply_loss(gt_land / img_size, render_lmks / img_size, 50) # 
        pix_loss = 100 * mseloss(masked_render, masked_gt)
        similarity_loss = cos_sim_costraint(shape_vec_cos_sim(shape_vec))


        latent_l = 100 * latent_loss(out)
        shape_norm = shape_vec.norm(p=1, dim=-1).mean()
        tex_norm = tex.norm(p=1, dim=-1).mean()
        norm_loss = 0.005 * (shape_norm + tex_norm)
        mesh_smooth_loss = mesh_normal_loss = torch.tensor([0])
        
        losses = {}
        if e<POS_EPOCH:
            losses['land_loss_2d'] = land_loss
            losses['latent_l'] = latent_l
       
            
        
        else: 
            
            mesh_smooth_loss = 10 * mesh_laplacian_smoothing(meshes)
            mesh_normal_loss = 2 * mesh_normal_consistency(meshes)
            losses['pix_loss'] = pix_loss
            losses['land_loss_2d'] = land_loss
            losses['latent_l'] = latent_l
            losses['pix_loss'] = pix_loss
            losses['norm_loss'] = norm_loss
            losses['contour_loss'] = contour_loss
            losses['similarity_loss'] = similarity_loss
            losses['mesh_smooth_loss'] = mesh_smooth_loss
            losses['mesh_normal_loss'] = mesh_normal_loss
         

        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
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
        if e in log_epoch and rand[0]<0.1: 
            for k, (img1, img2, msk1, msk2,dtp) in enumerate(zip(masked_render.detach().cpu(), masked_gt.detach().cpu(),  pred_masks.detach().cpu(), gt_masks.detach().cpu(),depth.detach().cpu())):
                msk1.unsqueeze(0)
                for land in render_lmks[k].detach().cpu().long():
                    land = torch.clamp(land,0,256-1)
                    img1[:,land[1]-1:land[1]+1,land[0]-1:land[0]+1]=torch.tensor([0,0,1]).unsqueeze(-1).unsqueeze(-1)
                    #print(r_msk.shape)
                img1 = img1 + (1-msk1)
                msk2.unsqueeze(0)
                for land in gt_land[k].detach().cpu().long():
                    land=torch.clamp(land,0,256-1)
                    
                    img2[:,land[1]-1:land[1]+1,land[0]-1:land[0]+1]=torch.tensor([0,0,1]).unsqueeze(-1).unsqueeze(-1)
                    #img2[:,land[1],land[0]]=torch.tensor([0,1,0])
                img2 = img2 + (1-msk2)
                save_image(torch.cat([img1, img2], 2), epoch_train_path+'/%d.jpg'%(i*batch_size+k))
                np.save(epoch_train_path+'/depth%d.npy'%(i*batch_size+k),(dtp*msk2).numpy())
                depth_display = (dtp*msk2).numpy()
                depth_display[depth_display<8.5] = math.nan
                plt.imshow(depth_display[0], cmap='hot', interpolation='nearest')
                
                cb = plt.colorbar() 


                plt.savefig(epoch_train_path+'/depth%d.jpg'%(i*batch_size+k),bbox_inches='tight')
                plt.cla()
                cb.remove() 
            for k, (out_, tex_, shape_vec_) in enumerate(zip(out, tex, shape_vec)):
                latent_f.writelines(['\n', str(i*batch_size+k), ' latent:\t']+['{:.4f} '.format(_) for _ in out_.cpu().tolist()]+ \
                            ['\n', str(i*batch_size+k), ' texture\t']+['{:.4f} '.format(_) for _ in tex_.cpu().tolist()]+ \
                            ['\n', str(i*batch_size+k), ' shape\t\t']+['{:.4f} '.format(_) for _ in shape_vec_.cpu().tolist()])
            for k, (verts, faces, tex_map) in enumerate(zip(meshes.verts_list(), meshes.faces_list(), tex_maps)):
                save_obj(os.path.join(epoch_train_path, 'mesh_%05d.obj' % (i*batch_size+k)), verts, faces, 
                            faces_uvs=decoder.faces_uvs, verts_uvs=decoder.verts_uvs, texture_map=tex_map)
        # break
    if e in log_epoch: 
        latent_f.close()

    
    print('train_loss: {:.4f} time: {:.4f}'.format(avg_loss/cnt, time.time() - start))

    if avg_loss/cnt<best_loss:
        torch.save(encoder.state_dict(), os.path.join(save_path, "resnet_encoder_train.pt"))
        best_loss = avg_loss/cnt

    if e in s2m_epoch:
        s2m_save_path = os.path.join(save_path,'epoch_%d/s2m/'%e)
        os.makedirs(s2m_save_path, exist_ok=True)    
        s2m = s2m_test(cfg, s2m_save_path)
        torch.save(encoder.state_dict(), os.path.join(save_path, "resnet_encoder_%d.pt"%e))    
        torch.cuda.empty_cache()
        last_s2m = s2m.test(encoder.eval(),fcrn_model.eval())

    scheduler.step()
    encoder.eval()
    test_cnt = 0
    avg_test_loss = 0
    with torch.no_grad():
        for i,data in enumerate(testloader):
            bs = data[0].shape[0]
            masked_gt=(data[0]*data[2]).to(device)
            gt_masks = data[2].to(device)

            with torch.no_grad():
                depth, frcn_feat = fcrn_model(masked_gt)
            out, tex, shape_vec = encoder(masked_gt,frcn_feat)
            
            render_images, project_verts, meshes, tex_maps, _, _ = decoder(out, tex, shape_vec)
            render_mask = render_images[..., 3].clone()
            render_images = render_images[..., :3] * render_images[..., 3:4]
            render_images = render_images.permute(0,3,1,2) # reshape to B, C, H, W

            render_lmks = project_verts[:,land_mark,:2]
            
            pred_masks = get_face_mask(img_size, render_lmks.detach().cpu().long().numpy())
            masked_render = render_images * pred_masks.to(device)

            gt_land = data[3].to(device)
            
            lmk_bbox_length = torch.sqrt(torch.sum((torch.max(gt_land, dim=1).values - torch.min(gt_land, dim=1).values)**2,dim=1))
            land_loss = 10 * torch.mean(torch.mean(torch.sqrt(torch.sum((gt_land - render_lmks)**2,dim=2)), dim=1) / lmk_bbox_length)
            contour_loss = 100 * ply_loss(gt_land / img_size, render_lmks / img_size, 50) # 
            pix_loss = 100 * mseloss(masked_render, masked_gt)
            similarity_loss = (shape_vec_cos_sim(shape_vec) + 1)
            
            latent_l = 100 * latent_loss(out)
            shape_norm = shape_vec.norm(p=1, dim=-1).mean()
            tex_norm = tex.norm(p=1, dim=-1).mean()
            norm_loss = 0.005 * (shape_norm + tex_norm)
            mesh_smooth_loss = mesh_normal_loss = torch.tensor([0])
            losses = {}
            if e<POS_EPOCH:
                losses['land_loss_2d'] = land_loss
                losses['latent_l'] = latent_l
           
            else: 
                
                mesh_smooth_loss = 10 * mesh_laplacian_smoothing(meshes)
                mesh_normal_loss = 2 * mesh_normal_consistency(meshes)
                losses['pix_loss'] = pix_loss
                losses['land_loss_2d'] = land_loss
                losses['latent_l'] = latent_l
                losses['pix_loss'] = pix_loss
                losses['norm_loss'] = norm_loss
                losses['contour_loss'] = contour_loss
                losses['similarity_loss'] = similarity_loss
                losses['mesh_smooth_loss'] = mesh_smooth_loss
                losses['mesh_normal_loss'] = mesh_normal_loss
            

            all_loss = 0.
            losses_key = losses.keys()
            for key in losses_key:
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss


            avg_test_loss += all_loss.item()
            test_cnt += 1
        
            loss_display = {}
            
            for k,v in losses.items():
                loss_display[k] = v.item()
            if i % 20 == 0:
                print(('Iter: {}|{} ').format(i, len(testloader)), loss_display)
            # exit()
            if (e%10==0 and e!=begin_epoch) or first_epoch: 
                torch.save(encoder.state_dict(), os.path.join(save_path, "resnet_encoder_%d.pt"%e))

            if e in log_epoch: 

                for k, (img1, img2, msk1, msk2,dtp) in enumerate(zip(masked_render.detach().cpu(), masked_gt.detach().cpu(),  pred_masks.detach().cpu(), gt_masks.detach().cpu(),depth.detach().cpu())):
                    msk1.unsqueeze(0)
                    for land in render_lmks[k].detach().cpu().long():
                        land = torch.clamp(land,0,256-1)
                        img1[:,land[1]-1:land[1]+1,land[0]-1:land[0]+1]=torch.tensor([0,0,1]).unsqueeze(-1).unsqueeze(-1)
                        #print(r_msk.shape)
                    img1 = img1 + (1-msk1)
                    msk2.unsqueeze(0)
                    for land in gt_land[k].detach().cpu().long():
                        land=torch.clamp(land,0,256-1)
                        
                        img2[:,land[1]-1:land[1]+1,land[0]-1:land[0]+1]=torch.tensor([0,0,1]).unsqueeze(-1).unsqueeze(-1)
                        #img2[:,land[1],land[0]]=torch.tensor([0,1,0])
                    img2 = img2 + (1-msk2)
                    save_image(torch.cat([img1, img2], 2), epoch_test_path+'/%d.jpg'%(i*batch_size+k))
                    np.save(epoch_test_path+'/depth%d.npy'%(i*batch_size+k),(dtp*msk2).numpy())
                    depth_display = (dtp*msk2).numpy()
                    depth_display[depth_display<8.5] = math.nan
                    plt.imshow(depth_display[0], cmap='hot', interpolation='nearest')
                    
                    cb = plt.colorbar() 


                    plt.savefig(epoch_test_path+'/depth%d.jpg'%(i*batch_size+k),bbox_inches='tight')
                    plt.cla()
                    cb.remove() 
                for k, (out_, tex_, shape_vec_) in enumerate(zip(out, tex, shape_vec)):
                    latent_f_test.writelines(['\n', str(i*batch_size+k), ' latent:\t']+['{:.4f} '.format(_) for _ in out_.cpu().tolist()]+ \
                                ['\n', str(i*batch_size+k), ' texture\t']+['{:.4f} '.format(_) for _ in tex_.cpu().tolist()]+ \
                                ['\n', str(i*batch_size+k), ' shape\t\t']+['{:.4f} '.format(_) for _ in shape_vec_.cpu().tolist()])
                for k, (verts, faces, tex_map) in enumerate(zip(meshes.verts_list(), meshes.faces_list(), tex_maps)):
                    save_obj(os.path.join(epoch_test_path, 'mesh_%05d.obj' % (i*batch_size+k)), verts, faces, 
                                faces_uvs=decoder.faces_uvs, verts_uvs=decoder.verts_uvs, texture_map=tex_map)
            
    if avg_test_loss/test_cnt < best_test_loss:
        torch.save(encoder.state_dict(), os.path.join(save_path, "resnet_encoder.pt"))
        best_test_loss = avg_test_loss/test_cnt
    print('test_loss:',avg_test_loss/test_cnt)
    first_epoch = False

print('best_test_loss:', best_test_loss)


torch.cuda.empty_cache()
s2m = s2m_test(cfg,save_path)
last_s2m = s2m.test(encoder.eval(),fcrn_model.eval())
print('last s2m:',last_s2m,'mean:',torch.mean(last_s2m))
print(save_path)



