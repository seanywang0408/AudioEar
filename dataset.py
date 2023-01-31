import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import glob
import os
from PIL import Image, ImageOps
from torchvision import transforms
import cv2
import math
# from utils import data_preprocess
import math
import matplotlib.pyplot as plt
from pytorch3d.io import IO
def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))
import json


def rand_rotate(img,key_pts,angle = None):
    diff = key_pts[0]-key_pts[19]
    angel = np.arctan(diff[1]/diff[0])

    ori_angel=np.rad2deg(angel)
    if ori_angel<0:
        ori_angel = ori_angel+180

    if not angle:
        rand_angel = np.random.randint(30,120)
    else: 
        rand_angel = angle
    rot_angel = ori_angel - rand_angel
    (h, w) = img.shape[:2] 
    center = (h // 2, w // 2) 
    M = cv2.getRotationMatrix2D(center, rot_angel, 1.0) #12
    img_rot = cv2.warpAffine(img, M, (w, h)) #13
    theta = np.radians(-rot_angel)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    kps_rot = (np.matmul(R,(key_pts - center).T).T + center)#.astype(int)
    return img_rot, kps_rot

def img_crop(img, key_pts, pad):
    """input img:np.array
             key_pts:np.array
    """
    h, w = img.shape[0], img.shape[1]

    wmax = int(np.max(key_pts[:,0]))
    wmin = int(np.min(key_pts[:,0]))
    hmax = int(np.max(key_pts[:,1]))
    hmin = int(np.min(key_pts[:,1])) 

    sel_center_y=int((hmax+hmin)/2)
    sel_center_x=int((wmax+wmin)/2)
    
    crop_w=max(wmax-wmin,hmax-hmin)+pad
    if crop_w > min(h,w):
        return img, key_pts
    crop_h=crop_w
    
    crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0)) #if left is out of bound, set left to 0
    crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0)) # if down is out of bound set down to 0
    
    diff_x = max(crop_x1 + crop_w - w, int(0)) # if right is out of bound, move left bound to have right to be img_width-1
    crop_x1 -= diff_x 
    diff_y = max(crop_y1 + crop_h - h, int(0)) # if down is out of bound ...
    crop_y1 -= diff_y
    crop_img = np.copy(img[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w])
    key_pts = key_pts - [crop_x1,crop_y1]
    return crop_img, key_pts

# def data_preprocess(crop_img, key_pts, input_size):

#     """return type: mask:tensor
#                     crop_img: ndarray
#     """
#     #change gray scale image
#     # if len(crop_img.shape) == 2:
#     #     crop_img = np.stack((crop_img,)*3, axis=-1)
        
#     #normalize landmark
#     normalized_key = key_pts/(np.array(crop_img.shape[::-1][1:]))
    
#     #normalize image
#     crop_img = crop_img/255
    
#     #Resize, ToTensor, Normalize
#     crop_img = cv2.resize(crop_img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)

#     #build mask
#     mask = np.zeros((input_size,input_size), np.uint8)
    
#     resize_keypoints = (normalized_key*input_size).astype(int)
    
#     cv2.fillPoly(mask, [np.concatenate([resize_keypoints[:20,:], resize_keypoints[39:40,:], resize_keypoints[38:39,:], resize_keypoints[35:36,:]], axis=0)], (1))
#     mask = torch.from_numpy(mask).unsqueeze(-1)
#     mask = mask.permute(2,0,1)
#     return crop_img, mask, normalized_key



class EarDataLoader(Dataset):
    def __init__(self, root, pad=0, train=True,input_size=256):
        self.root = root
        if train:
            self.datalist = glob.glob(os.path.join(self.root,"train/*.png"),)
        else:
            self.datalist = glob.glob(os.path.join(self.root,"test/*.png"),)
        self.imagelist=[]
        self.pad=pad
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
       
        self.input_size=input_size
        self.train = train
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        
        key_pts = read_pts(self.datalist[index][:-3]+'pts')

        img = Image.open(self.datalist[index])
        img = np.array(img)
    
        #crop image to keep only the ear, key points also shifted
        crop_img,key_pts=img_crop(img,key_pts,self.pad)
        
        
        return input_tensor, normalized_key, mask, normalized_key*self.input_size


class EarExtendDataLoader(Dataset):
    def __init__(self, root, train, input_size,mask_cat = []):
        self.root = os.path.join(root,'labeled')
        with open(os.path.join(root,'split.json')) as f:
            split = json.load(f)
        if train:
            self.datalist = []
            for x in split["train"]:
                remove_flag = False
                instance_dir = os.path.join(self.root,x)
                with open(instance_dir,'r') as f:
                    anno = json.load(f)
                    for cat in mask_cat:
                        if anno['flags'][cat]:
                            remove_flag = True
                if not remove_flag:
                    self.datalist.append(instance_dir)
            #self.datalist =[os.path.join(self.root,x) for x in split['train']]
        else:
            self.datalist = []
            for x in split["test"]:
                remove_flag = False
                instance_dir = os.path.join(self.root,x)
                with open(instance_dir,'r') as f:
                    anno = json.load(f)
                    for cat in mask_cat:
                        if anno['flags'][cat]:
                            remove_flag = True
                if not remove_flag:
                    self.datalist.append(instance_dir)
            # self.datalist =[os.path.join(self.root,x) for x in split['test']]
        self.imagelist=[]
        self.datalist.sort()
        print(len(self.datalist), root)
        self.jitter = transforms.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.2, hue=0)
        self.mask_root = os.path.join(root, 'mask')
        os.makedirs(self.mask_root, exist_ok=True)
        self.input_size=input_size
        self.train = train
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        
        key_pts = np.load(self.datalist[index][:-4]+'npy').astype(np.float32)
        img = Image.open(self.datalist[index][:-4]+'png')
        img_id = self.datalist[index][-10:-5]
        
        # filp horizontally if it is a left ear
        if key_pts[9,0] > key_pts[35,0]:
            img = ImageOps.mirror(img)
            key_pts[:,0] = -key_pts[:,0] + np.array(img).shape[0]
        
        if self.train:
            img = self.jitter(img)

        img = np.array(img, dtype=float)[:,:,:3]
        img = img / 255.
        normalized_key = key_pts / img.shape[1]
        # img = cv2.resize(img, (int(img.shape[0] / 2), int(img.shape[1] / 2)), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        # build mask
        mask_path = self.mask_root + '/' + img_id + '.npy'
        if not os.path.isfile(mask_path):
            mask = np.zeros((self.input_size, self.input_size), np.float32)
            resize_keypoints = (normalized_key*self.input_size).astype(int)
            cv2.fillPoly(mask, [np.concatenate([resize_keypoints[:20,:], resize_keypoints[39:40,:], resize_keypoints[38:39,:], resize_keypoints[35:36,:]], axis=0)], (1.))
            np.save(mask_path, mask)
        else:
            mask = np.load(mask_path).astype(np.float32)
        mask = mask[None]
        
        return img, normalized_key, mask, normalized_key*self.input_size
class EarCombineDataLoader(Dataset):
    ## use ffhq images to train 
    def __init__(self, root, train=True,input_size=256,saved_mask = False,dataset = 'both'):
        self.ibug_root = os.path.join(root,'ibug_edited')
        self.ffhq_root = os.path.join(root,'FFHQ_ears/labeled')
        with open(os.path.join(root,'combine_split.json')) as f:

            split = json.load(f)
        
        if dataset == 'both':
            if train:
                self.datalist =[os.path.join(self.ffhq_root,x) for x in split['train'] if x[0] != 't'] + [os.path.join(self.ibug_root,x) for x in split['train'] if x[0] == 't']

            else:
                self.datalist =[os.path.join(self.ffhq_root,x) for x in split['test'] if x[0] != 't'] + [os.path.join(self.ibug_root,x) for x in split['test'] if x[0] == 't']
        elif dataset == 'ffhq':
            if train:
                self.datalist =[os.path.join(self.ffhq_root,x) for x in split['train'] if x[0] != 't']

            else:
                self.datalist =[os.path.join(self.ffhq_root,x) for x in split['test'] if x[0] != 't']
        elif dataset =='ibug':
            if train:
                self.datalist =[os.path.join(self.ibug_root,x) for x in split['train'] if x[0] == 't']

            else:
                self.datalist =[os.path.join(self.ibug_root,x) for x in split['test'] if x[0] == 't']
        self.datalist.sort()
        print(self.datalist)
        self.imagelist=[]
        
        self.jitter = transforms.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.2, hue=0)
        self.mask_root = os.path.join(root, 'mask')
        os.makedirs(self.mask_root, exist_ok=True)
        self.input_size=input_size
        self.train = train
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        
        key_pts = np.load(self.datalist[index][:-3]+'npy').astype(np.float32)
        img = Image.open(self.datalist[index][:-3]+'png')
        img_id = self.datalist[index][-10:-5]
        
        # filp horizontally if it is a left ear
        if key_pts[9,0] > key_pts[35,0]:
            img = ImageOps.mirror(img)
            key_pts[:,0] = -key_pts[:,0] + np.array(img).shape[0]
        
        if self.train:
            img = self.jitter(img)

        img = np.array(img, dtype=float)[:,:,:3]
        img = img / 255.
        normalized_key = key_pts / img.shape[1]
        # img = cv2.resize(img, (int(img.shape[0] / 2), int(img.shape[1] / 2)), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        # build mask
        mask_path = self.mask_root + '/' + img_id + '.npy'
        if not os.path.isfile(mask_path):
            mask = np.zeros((self.input_size, self.input_size), np.float32)
            resize_keypoints = (normalized_key*self.input_size).astype(int)
            cv2.fillPoly(mask, [np.concatenate([resize_keypoints[:20,:], resize_keypoints[39:40,:], resize_keypoints[38:39,:], resize_keypoints[35:36,:]], axis=0)], (1.))
            np.save(mask_path, mask)
        else:
            mask = np.load(mask_path).astype(np.float32)
        mask = mask[None]
        
        return img, normalized_key, mask, normalized_key*self.input_size


class EarValImgLoader(Dataset):
    def __init__(self, root, input_size):
        self.root = root
        self.datalist = glob.glob(os.path.join(self.root,"*/left.jpg")) + glob.glob(os.path.join(self.root,"*/right.jpg"))
        self.datalist.sort()
        print('len of s2m dataset:', (len(self.datalist)))
        print(root)
        # for i in range(len(self.datalist)):
        #     print(i, self.datalist[i])
        self.input_size = input_size
        self.IO_3d = IO()
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        
        img_path = self.datalist[index]
        img = Image.open(img_path)
        json_path = img_path[:-3]+'json'
        pos_json_path = img_path[:-4]+'_pos.json'

        with open(json_path) as f:
            kp_dict = json.load(f)
        pt_list = []
        for cor in kp_dict['shapes']:
            pt = np.array(cor['points'])
            pt_list.append(pt)
        key_pts = np.concatenate(pt_list,axis = 0).astype(np.float32)
        with open(pos_json_path) as pos_f:
            pos_dict = json.load(pos_f)
        pos_h = pos_dict['pos']
        if key_pts[9,0] > key_pts[35,0]:
            img = ImageOps.mirror(img)
            key_pts[:,0] = -key_pts[:,0]+np.array(img).shape[0]
            pos = torch.tensor([pos_h[1], pos_h[0]-math.pi/2, -pos_h[2]])
        else:
            pos = torch.tensor([pos_h[1], -pos_h[0]-math.pi/2, pos_h[2]])
        img = np.array(img, dtype=float)[:,:,:3]
        img = img / 255.
        normalized_key = key_pts / img.shape[1]
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        # build mask
        mask = np.zeros((self.input_size, self.input_size), np.float32)
        resize_keypoints = (normalized_key*self.input_size).astype(int)
        cv2.fillPoly(mask, [np.concatenate([resize_keypoints[:20,:], resize_keypoints[39:40,:], resize_keypoints[38:39,:], resize_keypoints[35:36,:]], axis=0)], (1.))

        mask = mask[None]
        
        return img, normalized_key, mask, normalized_key*self.input_size, pos


    
class EarValLoader(Dataset):
    def __init__(self, root, train=True,input_size=256):
        self.root = root
        
        self.datalist = glob.glob(os.path.join(self.root,"*/left.jpg")) + glob.glob(os.path.join(self.root,"*/right.jpg"))
        self.datalist.sort()
        print('len of s2m dataset:', (len(self.datalist)))
        print(root)
        # for i in range(len(self.datalist)):
        #     print(i, self.datalist[i])
        self.input_size = input_size
        self.IO_3d = IO()

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        
        img_path = self.datalist[index]
        json_path = img_path[:-3]+'json'
        instance_path, direction = os.path.split(img_path)
        ply_path = img_path[:-3]+'ply'

        direction = direction[:-4]
        img = Image.open(img_path)

        input_pcd = self.IO_3d.load_pointcloud(ply_path)
        points, normals = input_pcd.points_list()[0], input_pcd.normals_list()[0]

        with open(json_path) as f:
            kp_dict = json.load(f)
        pt_list = []
        for cor in kp_dict['shapes']:
            pt = np.array(cor['points'])
            pt_list.append(pt)
        key_pts = np.concatenate(pt_list,axis = 0).astype(np.float32)
        if direction == 'left':

            img = ImageOps.mirror(img)
            points[:,2] = -points[:,2]
            normals[:,2] = -normals[:,2]
            key_pts[:,0] = -key_pts[:,0]+np.array(img).shape[0]

        img = np.array(img, dtype=float)[:,:,:3]
       
        normalized_key = key_pts / img.shape[1]
        ori_img_size = img.shape[0]
        img = img / 255.
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        
        
        
        img = torch.from_numpy(img)
        # build mask
        mask = np.zeros((self.input_size, self.input_size), np.float32)
        resize_keypoints = (normalized_key*self.input_size).astype(int)
        cv2.fillPoly(mask, [np.concatenate([resize_keypoints[:20,:], resize_keypoints[39:40,:], resize_keypoints[38:39,:], resize_keypoints[35:36,:]], axis=0)], (1.))

        mask = torch.from_numpy(mask[None])
        meta = {'image':img,'points':points,'normals':normals,'instance_id':os.path.join(os.path.split(instance_path)[1],direction),'mask':mask}
        return meta

class EarSytheticDataset(Dataset):
    def __init__(self, root, train=True, input_size=256, back_ground=True):
        if train:

            self.anno = np.load(os.path.join(root,'sythetic_anno.npz'),allow_pickle=True)['train'][()]
        else:
            self.anno = np.load(os.path.join(root,'sythetic_anno.npz'),allow_pickle=True)['test'][()]            
        self.root =root
        self.imagelist=[]
        self.jitter = transforms.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.2, hue=0)
        #self.mask_root = os.path.join(root, 'mask')
        #os.makedirs(self.mask_root, exist_ok=True)
        self.input_size=input_size
        self.train = train
        self.background = back_ground
    def __len__(self):
        return len(self.anno['image_index'])

    def __getitem__(self, index):
        
        key_pts = self.anno['kp_3d'][index]
        render_img =  Image.open(os.path.join(self.root,self.anno['image_index'][index],'render.png'))
        
        if self.background:
            img = Image.open(os.path.join(self.root,self.anno['image_index'][index],'render_background.jpg'))
            
        else:
            img = render_img
        
        if self.train:
            img = self.jitter(img)
    
        img = np.array(img, dtype=float)
        render_img = np.array(render_img,  dtype=float)
        mask = np.array(Image.open(os.path.join(self.root,self.anno['image_index'][index],'mask.jpg')), dtype=float)
        depth = np.load(os.path.join(self.root,self.anno['image_index'][index],'depth.npy')).transpose(1,2,0)
        
        render_img = render_img/255.
        img = img / 255.
        mask = mask/255.

        normalized_key_3d = key_pts / img.shape[1]
        # img = cv2.resize(img, (int(img.shape[0] / 2), int(img.shape[1] / 2)), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        render_img = cv2.resize(render_img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        depth = cv2.resize(depth, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # mask = np.transpose(render_img, (2, 0, 1))[3:4]
        
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))[0:1]
        
        normalized_key_3d = normalized_key_3d*self.input_size
        img = img[:3]
        # build mask
        
        
        
        mask
        
        out = np.concatenate([self.anno['rotation'][index],self.anno['xy'][index],
                        self.anno['f'][index],self.anno['light_color'][index],
                        self.anno['light_position'][index]])
        return {'image':img,'mask':mask,'kp_2d':key_pts[:,:2],
                'kp_3d':normalized_key_3d,'shape':self.anno['shape'][index],
                'texture':self.anno['texture'][index],
                'rotation':self.anno['rotation'][index],
                'xy':self.anno['xy'][index],'f':self.anno['f'][index],
                'light_color':self.anno['light_color'][index],'light_position':self.anno['light_position'][index],
                'out': out,
                'depth': depth,
                }
        