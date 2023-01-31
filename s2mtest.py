
import os
import time
import pickle

from pytorch3d.io.obj_io import load_obj
import _init_paths


import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from pytorch3d.io import load_ply, save_obj, save_ply
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer_distance, mesh_normal_consistency, mesh_laplacian_smoothing
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.renderer import look_at_rotation
from pytorch3d.ops import sample_points_from_meshes
from scipy.spatial import ConvexHull
from custom_loss import sample_points_from_polylines, custom_chamfer_distance
from dataset import EarValLoader
from models import Render, range_loss, ply_loss
from utils import get_face_mask, rgb2gray
from torch.autograd import Variable
from torchvision.utils import save_image
import math
import matplotlib.pyplot as plt
mseloss = nn.MSELoss()


rot_loss1 = range_loss(-math.pi/3, math.pi/3) # 绕从屏幕穿出的轴
rot_loss2 = range_loss(-1.5, -0.5) # 绕纵轴
rot_loss3 = range_loss(-math.pi/3, math.pi/3) # 绕横轴
f_loss = range_loss(0.5, 4) # scale
light_loss = range_loss(0, 80)

def latent_loss(out):
    return rot_loss2(out[:,1]) + f_loss(out[:,5]) 


class s2m_test(object):
    def __init__(self, cfg, save_path, img_size=256, shape_vec_value='pred'):
        self.data_path = cfg.s2m.s2m_data_path
        
        
        self.dataset = EarValLoader(self.data_path)

        self.ear_face_model_path = cfg.s2m.ear_face_model_path
        with open(cfg.model.pkl_path, 'rb') as f:
            ear_model = pickle.load(f)

        mu = torch.tensor(ear_model['Mean']).float()
        mu = mu.reshape(-1, 3)
        mu = mu - mu.mean(0)
        mu = mu.reshape(-1, 1)
        self.mu = mu
        self.faces = torch.from_numpy(ear_model['Trilist'])
        self.U = torch.tensor(ear_model['Eigenvectors']).float()
        self.V = torch.tensor(ear_model['EigenValues']).float()
        front_mesh = load_obj(self.ear_face_model_path)
        self.front_faces = front_mesh[1].verts_idx
        WS = pd.read_excel(cfg.model.land_mark_path)
        WS_np = np.array(WS)
        self.land_mark = torch.tensor(WS_np[WS_np[:,1]>0][:,1]).long()
        

        ## training settings
        self.sample_num = cfg.s2m.pts_num
        self.n_iters = cfg.s2m.total_epoch
        self.keypoint_epoch = cfg.s2m.keypoint_epoch
        self.mix_epoch = cfg.s2m.mix_epoch
        self.img_size = img_size

        self.save_path = save_path
        self.ear_model_key_points_ind = torch.tensor([978, 2, 75, 688]).cuda() # [top, bottom, left, right] [978, 2, 75, 688]
        self.shape_vec_value = shape_vec_value
        self.cfg = cfg
        
    def test(self, encoder,fcrn = None):
        print('###################')
        print('Start s2m testing')
        print('###################')
        device = next(encoder.parameters()).device
        #device = encoder.parameters().device
        #encoder = encoder.eval()
        
        self.land_mark = self.land_mark.to(device)
        self.front_faces = self.front_faces.to(device)
        self.mu = self.mu.to(device)
        self.faces = self.faces.to(device)
        self.U = self.U.to(device)
        self.V = self.V.to(device)
        self.decoder = Render(self.mu, self.V, self.U, self.faces, self.cfg.model)
        self.decoder = self.decoder.to(device)
        print_iters = [k*(self.n_iters//3) for k in range(1,4)]
        #print(print_iters)
        final_scan_2_mesh = torch.zeros(len(self.dataset))
        final_all_s2m = torch.zeros(len(self.dataset))
        final_pix_loss = torch.zeros(len(self.dataset))
        final_contour_loss = torch.zeros(len(self.dataset))
        final_land_loss = torch.zeros(len(self.dataset))
        final_latent_l = torch.zeros(len(self.dataset))
        final_norm_loss = torch.zeros(len(self.dataset))
        final_mesh_smooth_loss = torch.zeros(len(self.dataset))
        final_mesh_normal_loss = torch.zeros(len(self.dataset))
        final_filename = []
        for idx,data in enumerate(self.dataset):
           
            image = data['image'].unsqueeze(0).to(device)
            #gt_land = data['lmks'].unsqueeze(0).to(device)
            points = data['points'].to(device)
            # pos = data['pos'].to(device)
            mask = data['mask'].to(device)

            pt_idx = torch.randperm(len(points))[:self.sample_num]
            all_points = points
            points = points[pt_idx]

            normals = data['normals'].to(device)
            normals = normals[pt_idx]
            instance_save_path = os.path.join(self.save_path, data['instance_id'])
            os.makedirs(instance_save_path, exist_ok=True)

            masked_img = (image*mask)
            # save_image(masked_img[0],'tmp.jpg')
            
            with torch.no_grad():
                depth, fcrn_feat = fcrn(masked_img)
                out, tex, shape_vec = encoder(masked_img,fcrn_feat)
                # out[0,:3] = pos
            
            

            model = FitModel(ear_mu=self.mu, ear_eigenvectors=self.U, V=self.V, shape_vec=shape_vec, shape_vec_value=self.shape_vec_value).to(device)
            
            plys = Pointclouds(points=[points]).to(device)
            max_x, max_y, max_z = points.max(0)[1]
            min_x, min_y, min_z = points.min(0)[1]
            ply_key_points = torch.index_select(points, 0, torch.LongTensor([max_y, min_y, min_z, max_z]).to(device))
            ply_convex_hull = points[ConvexHull(points[:,1:].cpu().numpy()).vertices]
            ply_cvh_sample_points = sample_points_from_polylines(ply_convex_hull)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.075*12)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.keypoint_epoch, self.n_iters//2, self.n_iters-self.n_iters//4], gamma=0.1)


            save_ply(os.path.join(instance_save_path, 'a_ply.ply'), points.cpu(), normals.cpu())
            save_ply(os.path.join(instance_save_path, 'a_ply_keyp.ply'), ply_key_points.cpu())
            save_ply(os.path.join(instance_save_path, 'a_ply_cvh.ply'), ply_cvh_sample_points.cpu())
            np.save(os.path.join(instance_save_path, 'depth.npy'),depth.cpu().numpy())

            depth_display = (depth*mask).cpu().numpy()[0]
            depth_display[depth_display<8.5] = math.nan
            plt.imshow(depth_display[0], cmap='hot', interpolation='nearest')
            
            cb = plt.colorbar() 


            plt.savefig(instance_save_path+'/depth.jpg',bbox_inches='tight')
            plt.cla()
            cb.remove() 
            save_image(mask[0],os.path.join(instance_save_path, 'mask.jpg'))
            
            min_scan_2_mesh = 100000
            if self.shape_vec_value == 'optim':
                model.shape_vec.requires_grad = False
            for iteration in range(self.n_iters):
                
                
                optimizer.zero_grad()

                verts, R, T = model()
                meshes = Meshes(verts=[verts], faces=[self.front_faces])
                mesh_key_points = torch.index_select(verts, 0, self.ear_model_key_points_ind)

                if self.shape_vec_value == 'optim' and iteration==(self.n_iters*2)//5:
                    model.shape_vec.requires_grad = True
                
                if iteration<self.n_iters//3:
                    
                    loss_key_points = (mesh_key_points - ply_key_points).norm(dim=-1, p=2).sum()
                   
                    loss_all_kp = loss_verts = loss_normals = torch.tensor(0)
                    loss_s2m = torch.tensor(0)

                elif iteration>=self.n_iters//3 and iteration<(self.n_iters*2)//5 :

                    sample_points, sample_normals = sample_points_from_meshes(meshes, return_normals=True)
                    _, loss_verts, _, loss_normals = custom_chamfer_distance(sample_points, points.unsqueeze(0), x_normals=sample_normals, y_normals=normals.unsqueeze(0))
                    loss_normals = loss_normals * 10

                    mesh_convex_hull = verts[ConvexHull(verts[:,1:].cpu().detach().numpy()).vertices]
                    mesh_cvh_sample_points = sample_points_from_polylines(mesh_convex_hull)        
                    loss_all_kp, _ = chamfer_distance(mesh_cvh_sample_points.unsqueeze(0), ply_cvh_sample_points.unsqueeze(0))
                    loss_key_points = torch.tensor(0)
                    loss_s2m = torch.tensor(0)
                else:
                    sample_points, sample_normals = sample_points_from_meshes(meshes, return_normals=True)
                    _, loss_verts, _, loss_normals = custom_chamfer_distance(sample_points, points.unsqueeze(0), x_normals=sample_normals, y_normals=normals.unsqueeze(0))

                    loss_normals = 10* loss_normals

                    loss_all_kp = torch.tensor(0)

                    loss_key_points = torch.tensor(0)

                    plys = Pointclouds(points=[points]).cuda()
                    loss_s2m = scan2mesh_distance(meshes, plys)
                    if loss_s2m < min_scan_2_mesh:
                        min_scan_2_mesh = loss_s2m#loss_verts
                        min_iter = iteration
                        torch.save(model.state_dict(), os.path.join(instance_save_path, 'best_model.dat'))
                    final_scan_2_mesh[idx] = min_scan_2_mesh
                    
                loss =(loss_all_kp + loss_key_points + loss_verts + loss_normals  + loss_s2m )

                loss.backward()
                optimizer.step()
                scheduler.step()
                if iteration in print_iters or iteration==self.n_iters-1:
                    print('Sample {} iter: {}/{} \t loss: {:.4f} \t loss_all_kp: {:.4f} \t loss_verts {:.4f} \t loss_normals {:.4f} \t loss_s_2m: {:.4f}'.format(
                        idx, iteration, self.n_iters, loss.item(), loss_all_kp.item(), loss_verts.item(), loss_normals.item(), loss_s2m.item()))
                    save_obj(os.path.join(instance_save_path, 'mesh_%05d.obj' % iteration), verts, self.faces.to(device))
                
                    save_ply(os.path.join(instance_save_path, 'mesh_keyp_%05d.ply' % iteration), mesh_key_points.cpu())
            
            final_filename.append(data['instance_id'])    
            model.load_state_dict(torch.load(os.path.join(instance_save_path, 'best_model.dat')))
            verts, R, T = model()
            meshes = Meshes(verts=[verts], faces=[self.front_faces])
            all_plys = Pointclouds(points=[all_points]).to(device)
            final_all_s2m[idx] = scan2mesh_distance(meshes, all_plys)
            print('Sample {} {:.4f}'.format(idx, final_all_s2m[idx]))
                  
        #print(final_filename) 
        dict = {'instance': final_filename, 's2m': final_scan_2_mesh.tolist(), 'all_s2m': final_all_s2m.tolist(), 
                'final_pix_loss':final_pix_loss.tolist(), 'final_contour_loss':final_contour_loss.tolist(), 'final_land_loss':final_land_loss.tolist(), 
                'final_latent_l':final_latent_l.tolist(), 'final_norm_loss':final_norm_loss.tolist(), 'final_mesh_smooth_loss':final_mesh_smooth_loss.tolist(),
                'final_mesh_normal_loss':final_mesh_normal_loss.tolist()}
                
        df = pd.DataFrame(dict)
        df = df.sort_values(by='instance')
        df.loc[len(final_filename)] = ['mean',torch.mean(final_scan_2_mesh).item(), torch.mean(final_all_s2m).item(), torch.mean(final_pix_loss).item(), 
                                            torch.mean(final_contour_loss).item(), torch.mean(final_land_loss).item(), torch.mean(final_latent_l).item(), 
                                            torch.mean(final_norm_loss).item(), torch.mean(final_mesh_smooth_loss).item(), torch.mean(final_mesh_normal_loss).item(), 
                                            ]
        df.to_csv(os.path.join(self.save_path,'s2mresult.csv'))
        #result = list(zip(final_filename,final_scan_2_mesh))   
        print('mean', torch.mean(final_scan_2_mesh).item(), torch.mean(final_all_s2m).item())
        print(final_scan_2_mesh)
        print(final_all_s2m)
        print(df)
        return final_scan_2_mesh
    
    
def scan2mesh_distance(meshes: Meshes, pcls: Pointclouds):
    '''
    from pytorch3d.loss.point_mesh_face_distance
    '''
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)
    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()
    # print(points_first_idx, tris_first_idx, max_points)
    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    
    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_face = point_to_face * weights_p
    point_dist = point_to_face.sum() / N
    return point_dist


class FitModel(nn.Module):
    def __init__(self, ear_mu, ear_eigenvectors, V, shape_vec, shape_vec_value):
        super().__init__()
        self.cam_pos = nn.Parameter(torch.tensor((0., 0, 0)))
        self.look_at = nn.Parameter(torch.tensor((0., 0, 1.)))
        self.up = nn.Parameter(torch.tensor((0., 1., 0., )))
        self.scale_factor = nn.Parameter(torch.tensor(120.))
        self.register_buffer('ear_mu', ear_mu)
        self.register_buffer('ear_eigenvectors', ear_eigenvectors)
        self.register_buffer('ear_eigenvalues', V)

        if shape_vec_value=='optim':
            self.shape_vec = nn.Parameter(torch.zeros(1, ear_eigenvectors.shape[1]).to(ear_mu.device))
        elif shape_vec_value=='avg':
            self.register_buffer('shape_vec', torch.zeros(1, ear_eigenvectors.shape[1]).to(ear_mu.device))
        else:
            self.register_buffer('shape_vec', shape_vec)

    def forward(self,):
       
        verts = self.ear_mu + self.ear_eigenvectors.mm((self.shape_vec*self.ear_eigenvalues).permute(1,0))
        verts = verts.view(-1, 3)#.detach()
        verts = max(0, self.scale_factor) * verts
        R = look_at_rotation(self.cam_pos[None, :], at=self.look_at[None, :], up=self.up[None,:], device=verts.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.cam_pos[None, :, None])[:, :, 0]   # (1, 3)
        R, T = R.squeeze(0), T.squeeze(0)
        verts = verts.mm(R) + T
        
        return verts, R, T





        
        