import torch
import torchvision.models as models
import math
import torch.nn as nn
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    BlendParams
    
)
from pytorch3d.transforms import euler_angles_to_matrix
from torchvision.utils import save_image
import numpy as np
import cv2
import pytorch3d


def build_mesh(verts,faces,verts_uvs,faces_uvs,tex):
    # print(verts.shape, faces.shape, verts_uvs.shape, faces_uvs.shape, tex.shape)


    bs = verts.shape[0]

    n_device = verts.device
    
    batch_verts_uvs=torch.cat(bs*[verts_uvs.unsqueeze(0)],dim=0).to(n_device)
    
    batch_faces=torch.cat(bs*[faces.unsqueeze(0)],dim=0).to(n_device)
    batch_faces_uvs=torch.cat(bs*[faces_uvs.unsqueeze(0)],dim=0).to(n_device)

    textures = TexturesUV(tex,batch_faces_uvs,batch_verts_uvs)

    mesh = Meshes(
        verts,   
        batch_faces, 
        textures
    )
    
    return mesh


def build_renderer(latent, img_size, device):
    
    R, T = look_at_view_transform(10, 0, math.pi/2,degrees=False)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # print(R.shape, T.shape)
    raster_settings = RasterizationSettings(
        image_size=img_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 

    )
    
    #lights = PointLights(ambient_color=((1., 1., 1.), ), diffuse_color=((0.5, 0.5, 0.5), ), specular_color=((0.2,0.2,0.2),),  device=device, location=latent[:,15:18],)
    lights = PointLights(ambient_color=latent[:,6:9], diffuse_color=latent[:,9:12], specular_color=latent[:,12:15], location=latent[:,15:18],  device=device)
    # lights = DirectionalLights(ambient_color=latent[:,6:9], diffuse_color=latent[:,9:12], specular_color=latent[:,12:15], device=device)
    materials = Materials(ambient_color=((1, 1, 1), ), 
                      diffuse_color=((1, 1, 1), ),
                      specular_color=((1, 1, 1), ), 
                      shininess=64,
                     device=device)
    # materials = Materials(ambient_color=latent[:,18:21], 
    #                   diffuse_color=latent[:,21:24],
    #                   specular_color=latent[:,24:27], 
    #                   shininess=latent[:,27],
    #                  device=device)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        # shader=SoftPhongShader(
        #     device=device, 
        #     cameras=cameras,
        #     lights=lights,
        #     materials=materials,
        # )
        shader=HardPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights,
            materials=materials,
        )
    )
    rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
                            
    return renderer, cameras, rasterizer


def deform_scale_rotation(laten,shape_vector,mu, V, U):
    """
    Inputs:
        shape_vector: B x 240
        V: 236
        U: 8400 x 236

    """
    bs = shape_vector.shape[0]
    device=shape_vector.device
    
    rot = laten[:,:3]
    rot_matrix = euler_angles_to_matrix(rot,'XYZ')

    xy_trans = laten[:,3:5]
    f = laten[:,5]
   

    #ear_shape = shape_vector
    
    mod_V = shape_vector*V
    #mod_V=V
    verts = (mu +torch.matmul(U,mod_V.unsqueeze(-1))).squeeze()
    
    verts = verts.reshape(bs,2800,3)
    center = (torch.min(verts,1).values+torch.max(verts,1).values)/2
    verts = verts-center.unsqueeze(1)
    verts[:,:,0] = -verts[:,:,0]
    verts[:,:,2] = -verts[:,:,2]
    
    # verts max: [0.1184, 0.2372, 0.1707],
    # verts min: [-0.1184, -0.2372, -0.1707]
    
    
    verts = verts*(f.unsqueeze(-1).repeat(1,3).unsqueeze(1))
    
    verts = (torch.matmul(rot_matrix.to(device),verts.permute(0,2,1))).permute(0,2,1)
    verts[:,:,1:] = verts[:,:,1:]+(xy_trans*f.unsqueeze(-1)).to(device).unsqueeze(1)
    return verts
    

def save_rendered(images, land_mark, batch_size, batch_index, save_path,img_size):
    for k,img in enumerate(images.detach().cpu()):
        img1 = img.permute(2,0,1)[:3,:,:]
        for land in land_mark[k].detach().cpu():
            land=torch.clamp(land,0,img_size-1).long()
            img1[:,land[1],land[0]]=1
        save_image(img1, save_path+'/%d.jpg'%(batch_index*batch_size+k))




def get_face_mask(img_size, land_mark):
    """
    derive mask for images
    Args:
        images: 
        land_mark: first 20 are the border
    """
    mask_list = []
    land_mark = land_mark.astype(int)
    # land_mark = land_mark.detach().cpu().long().numpy()
    for ldm in land_mark:
        ldm = np.clip(ldm, 0, img_size)
        mask = np.zeros((img_size, img_size), dtype=np.float32)
        # print(np.concatenate([ldm[:20,:], ldm[39:40,:], ldm[38:39,:], ldm[35:36,:]], axis=0).shape)
        cv2.fillPoly(mask, [np.concatenate([ldm[:20,:], ldm[39:40,:], ldm[38:39,:], ldm[35:36,:]], axis=0)], (1))
        mask_list.append(torch.from_numpy(mask[None]))
    return torch.stack(mask_list, dim=0)

def get_two_masks(img_size, land_mark):
    """
    derive mask for images
    Args:
        images: 
        land_mark: first 20 are the border
    """
    mask_list = []
    inner_list = []
    land_mark = land_mark.detach().cpu().long().numpy()
    for ldm in land_mark:
        ldm = np.clip(ldm, 0, img_size)
        mask = np.zeros((img_size, img_size), dtype=np.float32)
        inner_mask = np.zeros((img_size, img_size), dtype=np.float32)
        cv2.fillPoly(mask, [np.concatenate([ldm[:20,:], ldm[39:40,:], ldm[38:39,:], ldm[35:36,:]], axis=0)], (1))
        cv2.fillPoly(inner_mask, [ldm[35:50]], (1))
        mask_list.append(torch.from_numpy(mask[None]))
        inner_list.append(torch.from_numpy(inner_mask[None]))
    return torch.stack(mask_list, dim=0), torch.stack(inner_list, dim=0)



def rotate_to_horizon(landmarks,img_size = 256):
    diff = landmarks[:,0,:]-landmarks[:,19,:]
    angle = torch.atan(diff[:,1]/diff[:,0])
    
    angle[angle<0] = angle[angle<0]+math.pi
    c, s = torch.cos(angle).unsqueeze(-1), torch.sin(angle).unsqueeze(-1)
    R = torch.cat((torch.cat((c,-s),dim=-1).unsqueeze(-1),torch.cat((s,c),dim=-1).unsqueeze(-1)),dim=-1)
    

    rotate_pts = torch.matmul(R,(landmarks - img_size/2).transpose(1,2)).transpose(1,2)+img_size/2
    #R = np.array(((c, -s), (s, c)))
    return rotate_pts

def cal_diff(rotate_gt):
    #idx_blue = [1,2,3,5,6,7,8,9,10,11,12,13]
    #idx_lightb = [23,24,25,26,27,28,29,30,31,32,33,34]
    idx_blue = [1,2,5,8,10,12]
    idx_lightb = [23,24,26,29,31,33]
    idx_orange =[54,53,52,51,50]
    idx_greenpart1 = [49,48,47,46,45]

    idx_greenpart2 = [49, 46, 44,42,41]
    idx_greenpart3 = [35,35,36,37,38]

    outdiff_gt = rotate_gt[:,idx_blue,:] - rotate_gt[:,idx_lightb,:]

    indiff_gt = rotate_gt[:,idx_orange,:] - rotate_gt[:,idx_greenpart1,:]

    selfdiff_gt = rotate_gt[:,idx_greenpart2,:] - rotate_gt[:,idx_greenpart3,:]

    all_diff = torch.cat((outdiff_gt,indiff_gt,selfdiff_gt),dim=1)

    return all_diff


def sample_points_from_polylines_nobatch(polylines, num_samples, circle):
    '''
    polylines: Nx2 or Nx3
    circle: True or False
    '''
    assert type(circle) is bool
    with torch.no_grad():
        lengths = (polylines - polylines.roll(shifts=-1, dims=0)).norm(p=2, dim=-1)
        lengths = lengths + 1e-4 # robust
        if circle:
            start_idxs = lengths.multinomial(num_samples, replacement=True)
        else:
            start_idxs = lengths[:-1].multinomial(num_samples, replacement=True)
        end_idxs = (start_idxs+1)%len(polylines)
        w = torch.rand((num_samples, 1), device=polylines.device)
    return w * polylines[start_idxs] + (1 - w) * polylines[end_idxs]

def batch_sample(polylines,num_samples):
    line_list=[]
    for line in polylines:
        line_sample = sample_points_from_polylines_nobatch(line, num_samples, circle=False)
        line_list.append(line_sample)
    batch_line = torch.stack(line_list, dim=0)
    return batch_line


def contour_sampling(landmarks,num_samples):
    contour_1 = landmarks[:,0:20]
    contour_2 = landmarks[:,20:35]
    contour_3 = landmarks[:,35:50]
    contour_4 = landmarks[:,50:]

    # _,pts_1 = sample_points_from_polylines(contour_1,num_samples)
    # _,pts_2 = sample_points_from_polylines(contour_2,num_samples)
    # _,pts_3 = sample_points_from_polylines(contour_3,num_samples)
    # _,pts_4 = sample_points_from_polylines(contour_4,num_samples)
    pts_1 = batch_sample(contour_1,num_samples)
    pts_2 = batch_sample(contour_2,num_samples)
    pts_3 = batch_sample(contour_3,num_samples)
    pts_4 = batch_sample(contour_4,num_samples)
    return pts_1,pts_2,pts_3,pts_4

def rgb2gray(batch_img):
    # Input: B x H x W x 4

    gray = batch_img[:,0,:,:]* 0.299 + batch_img[:,1,:,:]* 0.587 +  batch_img[:,2,:,:]* 0.114
    return gray.unsqueeze(1)