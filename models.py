import torch
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.io import load_objs_as_meshes, load_obj


from utils import build_mesh, build_renderer,deform_scale_rotation,rotate_to_horizon, cal_diff, contour_sampling
from FLAME import FLAME, FLAMETex
import math
import pytorch3d.loss
from pytorch3d.ops import sample_points_from_meshes

class ResNet_concat(nn.Module):
    def __init__(self, two_res = True):
        super(ResNet_concat, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool


        self.fc = nn.Linear(in_features=512, out_features=34) #[xrot, yrot, zrot, x_trans,y_trans, f,light(3dim)]

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                    in_channels=512,
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.fc_tex  = nn.Linear(in_features=512, out_features=50)

        self.shape_bn1 = nn.BatchNorm1d(1024)
        self.shape_bn2 = nn.BatchNorm1d(512)

        self.shape_fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=236),
        )
        self.kp_fc = nn.Linear(in_features=512, out_features=55*2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        

        x = self.layer3(x)
       

        s_feat = self.layer4(x)
       
        x = self.avgpool(s_feat)
       

        #x = self.resnet(x)
        x = torch.flatten(x, 1)

        pos_vec = self.fc(x)
        tex = self.fc_tex(x)

        shape_vec = self.shape_fc(x)
        key_point = self.kp_fc(x)
        key_point = key_point.view(-1,55,2)

       
        s_feat = self.deconv(s_feat)
        return pos_vec, tex, shape_vec, key_point, s_feat


class Render(nn.Module):
    def __init__(self,mu, V, U, faces, model_cfg):
        super(Render, self).__init__()
        # If set receptive_keep=true, we won't perform downsample on stage4.

        self.deform_scale_rotation = deform_scale_rotation
        self.build_mesh = build_mesh
        self.mu = mu
        self.V = V
        self.U = U
        #self.faces = faces
        hera_verts, hera_faces, hera_aux = load_obj(model_cfg.hera_ear_path)
        self.faces = hera_faces.verts_idx
        self.faces_uvs = hera_faces.textures_idx
        self.verts_uvs = hera_aux.verts_uvs
        self.flametex = FLAMETex(model_cfg) 
        self.img_size = model_cfg.img_size

    def forward(self, laten, tex, shape_vec):

        verts = self.deform_scale_rotation(laten, shape_vec, self.mu, self.V, self.U)
        tex_map = self.flametex(tex).permute(0,2,3,1)
        mesh = self.build_mesh(verts, self.faces, self.verts_uvs, self.faces_uvs, tex_map)
        #  sample_points = sample_points_from_meshes(mesh,num_samples = 500)
        renderer, cameras, rasterizer = build_renderer(laten, self.img_size, verts.device)

        images = renderer(mesh)
        fragments = rasterizer(mesh)
        
        project_verts = cameras.transform_points(verts) # [-1, 1]
        
        project_verts = - project_verts * self.img_size/2 + self.img_size/2
        project_points = None
        # project_points = cameras.transform_points(sample_points) # [-1, 1]
        
        # project_points = - project_points * self.img_size/2 + self.img_size/2
        # if torch.isnan(project_verts).sum()!=0 or torch.isinf(images).sum()!=0:
        #     print('verts nan / images inf: ', torch.isnan(project_verts).sum(), torch.isinf(images).sum())
        #     print('laten', laten)
        #     print('shape', shape_vec)
        #     print('camera', cameras.get_full_projection_transform())
        return images, project_verts, mesh, tex_map, project_points, fragments.zbuf


class range_loss(nn.Module):
    def __init__(self, lower, upper):
        super(range_loss, self).__init__()
        self.min = lower
        self.max = upper
        
    def forward(self, x):
        mask_min = ((x-self.min)<0).float()
        mask_max = ((x-self.max)>0).float()
        return torch.mean((x-self.min)**2 * mask_min + (x-self.max)**2 * mask_max) 

def relative_ldm_loss(gt, pred,img_size = 256):
   

    rotate_gt = rotate_to_horizon(gt,img_size)-img_size/2
    rotate_pred = rotate_to_horizon(pred,img_size)-img_size/2

    rotate_gt = rotate_gt/torch.max(torch.abs(rotate_gt),dim=1).values.unsqueeze(1)

    rotate_pred = rotate_pred/torch.max(torch.abs(rotate_pred),dim=1).values.unsqueeze(1)

    diff_gt = cal_diff(rotate_gt)
    diff_pred = cal_diff(rotate_pred)

    land_loss = torch.mean(torch.sum(torch.sqrt(torch.sum((diff_gt-diff_pred)**2,dim=2)),dim=1)/(diff_gt.shape[1]*4))

    return land_loss


def rot_ldm_loss(gt, pred,img_size = 256):
   

    rotate_gt = rotate_to_horizon(gt,img_size)-img_size/2
    rotate_pred = rotate_to_horizon(pred,img_size)-img_size/2

    rotate_gt = rotate_gt/torch.max(torch.abs(rotate_gt),dim=1).values.unsqueeze(1)

    rotate_pred = rotate_pred/torch.max(torch.abs(rotate_pred),dim=1).values.unsqueeze(1)

    dn=torch.sqrt(torch.sum((torch.max(rotate_gt,dim=1).values-torch.min(rotate_gt,dim=1).values)**2,dim=1))
    land_loss = torch.mean(torch.sum(torch.sqrt(torch.sum((rotate_gt-rotate_pred)**2,dim=2)),dim=1)/(55*dn))

    return land_loss


def ldm_loss(gt_land, pred_land):
    land_loss = torch.mean(torch.sum(torch.sqrt(torch.sum((gt_land-pred_land)**2,dim=2)),dim=1)/(55*dn))
    return land_loss


from pytorch3d.loss import chamfer_distance
import time
def ply_loss(gt, pred,num_samples):
    # start = time.time()
    gt_1, gt_2, gt_3, gt_4 = contour_sampling(gt, num_samples)
    pred_1, pred_2, pred_3, pred_4 = contour_sampling(pred, num_samples)
    # print('sample', time.time()-start)
    # start = time.time()
    ply_loss = chamfer_distance(gt_1, pred_1) + chamfer_distance(gt_2,pred_2) + chamfer_distance(gt_3,pred_3) + chamfer_distance(gt_4,pred_4)
    # print('cd', time.time()-start)
    # print(ply_loss)
    return (ply_loss[0]+ply_loss[2]+ply_loss[4]+ply_loss[6]) / 4#, gt_1, gt_2, gt_3, gt_4, pred_1, pred_2, pred_3, pred_4




def rot_ply_loss(gt, pred,num_samples,img_size = 256):
    
    rotate_gt = rotate_to_horizon(gt,img_size)-img_size/2
    rotate_pred = rotate_to_horizon(pred,img_size)-img_size/2

    rotate_gt = rotate_gt/torch.max(torch.abs(rotate_gt),dim=1).values.unsqueeze(1)

    rotate_pred = rotate_pred/torch.max(torch.abs(rotate_pred),dim=1).values.unsqueeze(1)
    gt_1,gt_2,gt_3,gt_4 = contour_sampling(rotate_gt,num_samples)
    pred_1,pred_2,pred_3,pred_4 = contour_sampling(rotate_pred,num_samples)
    ply_loss = pytorch3d.loss.chamfer_distance(gt_1,pred_1)+pytorch3d.loss.chamfer_distance(gt_2,pred_2)+pytorch3d.loss.chamfer_distance(gt_3,pred_3)+pytorch3d.loss.chamfer_distance(gt_4,pred_4)

    return torch.mean(ply_loss[0])/(4*num_samples)


import torch.nn.functional as F
def shape_vec_cos_sim(shape_vec):
    norm_vec = F.normalize(shape_vec)
    bs = norm_vec.shape[0]
    cos_sim = torch.mm(norm_vec,torch.transpose(norm_vec, 0, 1))
    cos_sim_mean = (torch.sum(cos_sim)-bs) / (bs*(bs-1))
    #(torch.sum(cos_sim)-bs)/(bs*(bs-1))
    # cos_loss = range_loss(-2,0.5)

    return cos_sim_mean

class MAF_Extractor(nn.Module):
    ''' Mesh-aligned Feature Extrator
    As discussed in the paper, we extract mesh-aligned features based on 2D projection of the mesh vertices.
    The features extrated from spatial feature maps will go through a MLP for dimension reduction.
    '''

    def __init__(self, device=torch.device('cuda'), num_points = 500):
        super().__init__()

        self.device = device
        self.filters = []
        self.num_views = 1
        filter_channels = [256, 128, 64, 5]
        self.last_op = nn.ReLU(True) 

        for l in range(0, len(filter_channels) - 1):
            if 0 != l:
                self.filters.append(
                    nn.Conv1d(
                        filter_channels[l] + filter_channels[0],
                        filter_channels[l + 1],
                        1))
            else:
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))

            self.add_module("conv%d" % l, self.filters[l])
        
        self.im_feat = None
        self.cam = None
        self.fc =  nn.Linear(in_features=num_points*5, out_features=236)


    def reduce_dim(self, feature):
        '''
        Dimension reduction by multi-layer perceptrons
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            y = self._modules['conv' + str(i)](
                y if i == 0
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        y = self.last_op(y)

        y = y.view(y.shape[0], -1)
        return y

    def sampling(self, points, im_feat=None, z_feat=None):
        '''
        Given 2D points, sample the point-wise features for each point, 
        the dimension of point-wise features will be reduced from C_s to C_p by MLP.
        Image features should be pre-computed before this call.
        :param points: [B, N, 2] image coordinates of points
        :im_feat: [B, C_s, H_s, W_s] spatial feature maps 
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        if im_feat is None:
            im_feat = self.im_feat

        batch_size = im_feat.shape[0]

        
        point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2), align_corners=True)[..., 0]
        
        mesh_align_feat = self.reduce_dim(point_feat)
        return mesh_align_feat

    def forward(self, p, s_feat=None, cam=None, **kwargs):
        ''' Returns mesh-aligned features for the 3D mesh points.
        Args:
            p (tensor): [B, N_m, 3] mesh vertices
            s_feat (tensor): [B, C_s, H_s, W_s] spatial feature maps
            cam (tensor): [B, 3] camera
        Return:

            #mesh_align_feat (tensor): [B, C_p x N_m] mesh-aligned features
            mesh_align_feat after mlp to shape vector [B, 236]
        '''
        mesh_align_feat = self.sampling(p, s_feat)
        
        return self.fc(mesh_align_feat)




class ResNet_FCRN(nn.Module):
    def __init__(self,):
        super(ResNet_FCRN, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool


        self.layer1_conv = nn.Conv2d(64+16, 64, 3, 1, 1)
        self.layer2_conv = nn.Conv2d(64+32, 64, 3, 1, 1)
        self.layer3_conv = nn.Conv2d(128+64, 128, 3, 1, 1)
        self.layer4_conv = nn.Conv2d(256+128, 256, 3, 1, 1)

        self.fc = nn.Linear(in_features=512, out_features=34) #[xrot, yrot, zrot, x_trans,y_trans, f,light(3dim)]
        self.fc_tex  = nn.Linear(in_features=512, out_features=50)

        self.shape_fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(in_features=512, out_features=236),
        )
        
 
    def forward(self, x, frcn_feat):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape, frcn_feat[-1].shape)
        # print(x.shape)
        x = self.layer1_conv(torch.cat([x, self.maxpool(frcn_feat[-1])], 1)) # no downsample and channel addition in resnet.layer1
        x = self.layer1(x)
        # print(x.shape, frcn_feat[-2].shape) # 64, 32
        x = self.layer2_conv(torch.cat([x, frcn_feat[-2]], 1))
        x = self.layer2(x)
        # print(x.shape, frcn_feat[-3].shape) # 128, 64
        x = self.layer3_conv(torch.cat([x, frcn_feat[-3]], 1))
        x = self.layer3(x)
        # print(x.shape, frcn_feat[-4].shape) # 256, 128
        x = self.layer4_conv(torch.cat([x, frcn_feat[-4]], 1))
        x = self.layer4(x) # 512
        

        x = self.avgpool(x)
        x = torch.flatten(x, 1)


        pos_vec = self.fc(x)
        tex = self.fc_tex(x)
        shape_vec = self.shape_fc(x)
        

       
        return pos_vec, tex, shape_vec
