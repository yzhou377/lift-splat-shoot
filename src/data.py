"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob
import yaml 

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx, downsize, process_orig_img


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]
        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        return samples
    
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        orig_imgs=[]
        
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            orig_img= Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            
            orig_imgs.append(process_orig_img(downsize(orig_img,1/8)))
            imgs.append(normalize_img(img)) 
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(orig_imgs), torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)
        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    # For visualization only 
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        orig_img, imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3) # Difference is adding the lidar here. 
        binimg = self.get_binimg(rec)
        
        return orig_img, imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg, rec


class SegmentationData(NuscData):
    # For training and evaluation 
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        cams = self.choose_cams()
        orig_img, imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
        
        return orig_img, imgs, rots, trans, intrins, post_rots, post_trans, binimg, rec







class CarlaData(torch.utils.data.Dataset):
    def __init__(self, dataroot, is_train, data_aug_conf, grid_conf):
        # inheriting the parent class of dataset 
        
        
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.dataroot= dataroot
        self.metadata= self.load_metadata()

        # Returns a summary of all scenes with splits
        self.ixes= self.prepro()
        # Returns the grid size
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        torch.utils.data.Dataset.__init__(dataroot, is_train, data_aug_conf, grid_conf)

    def load_metadata(self):
        metadata_path= os.path.join(self.dataroot,'000000','metadata.yaml')
        with open(metadata_path) as file:
            content= yaml.load(file, Loader=yaml.FullLoader)
            return content 


    def prepro(self):
        """
        Return a list of all scenes, each scene is a dictionary with 'label' and 'inputs' for 6 cameras
        Incomplete data are elimited here.  
        DONE: scenes are incomplete on Mar.08, from Takeda in email. YZ. 
        """
        data_summary=[]
        if self.is_train:
            start_folder=0
            end_folder=69
        else:
            start_folder=70
            end_folder=99

        for folder in sorted(glob(os.path.join(self.dataroot, '0000*')))[start_folder:end_folder]:
            label_files=glob(os.path.join(folder, 'bev_label_000*'))
            bev_files= glob(os.path.join(folder, 'rgb_*'))
            for label_file in label_files:
                in_folder_index= label_file[-10:-4]
                inputs_file_names = [os.path.join(folder,"rgb_{}_{}.jpg".format(cam, in_folder_index)) for cam in self.data_aug_conf['cams']] 
                if all(item in bev_files for item in inputs_file_names):
                    data_summary.append({'label': label_file, 'folder': folder, 'in_folder_index': in_folder_index})
        return data_summary

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_intrinsic(self,cam):
        """
        Return intrinsic matrix of shape 
        Under the dataroot directory, there exists a calibration folder, with 6 extrinsics and 6 intrinsics
        Format confirmed to be in alignment with NuScenes
        """ 
        intrinsic=  np.fromstring(self.metadata['intrinsics'][cam], dtype=np.float, count=9).reshape(3,3)
        return torch.Tensor(intrinsic)


    def get_extrinsic(self,cam):
        """
        Return rotation of shape (3,3) and translatin of shape (3) in tensor format

        Under the dataroot directory, there exists a calibration folder, with 6 extrinsics and 6 intrinsics
        Format confirmed to be in alignment with NuScenes
        """
        extrinsic= np.fromstring(self.metadata['extrinsics'][cam], dtype=np.float, count=12).reshape(3,4)
        extrinsic[2,1]*=(-1)
        rot= torch.Tensor(extrinsic[0:3,0:3])
        tran = torch.Tensor(extrinsic[0:3,3])
        return rot, tran 


    def get_image_data(self, rec, cams):
        """
        Return the image normalized by tools.normalize
        """
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        orig_imgs=[]
        for cam in cams:
            imgname = os.path.join(rec['folder'], "rgb_{}_{}.jpg".format(cam, rec['in_folder_index']))
            img = Image.open(imgname)
            orig_img= Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            intrin = self.get_intrinsic(cam)
            rot, tran = self.get_extrinsic(cam)

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            
            #print(torch.Tensor(np.array(orig_img).T.shape))
            orig_imgs.append(process_orig_img(downsize(orig_img,1/8)))
            imgs.append(normalize_img(img)) 
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(orig_imgs), torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))


    def get_binimg(self, rec):
        """
        Return the groundtruth map in grid sizes 
        """
        label_file_path= rec['label']
        img = Image.open(label_file_path).convert('L')
        img = img.resize((self.nx[0],self.nx[0]))
        img_np= np.array(img)>0
        return torch.Tensor(img_np).unsqueeze(0)


    def choose_cams(self):
        """
        Choose cams if we choose camera drop off, only during training we choose to drop. 
        """
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams


    def __getitem__(self, index):
        """
        The main function to get all the information setup for training or validation 
        """
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        orig_img, imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
        
        return orig_img, imgs, rots, trans, intrins, post_rots, post_trans, binimg, rec

    def __len__(self):
        return len(self.ixes)

def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):

    if version=="CARLA":
        traindata= CarlaData(dataroot= dataroot, is_train=True, data_aug_conf= data_aug_conf,
                            grid_conf= grid_conf)
        valdata= CarlaData(dataroot= dataroot, is_train=False, data_aug_conf= data_aug_conf,
                            grid_conf= grid_conf)
        
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                                shuffle=True,
                                                num_workers=nworkers,
                                                drop_last=True,
                                                worker_init_fn=worker_rnd_init)
        valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                                shuffle=False,
                                                num_workers=nworkers)

    else:
        nusc = NuScenes(version='v1.0-{}'.format(version),
                        dataroot=os.path.join(dataroot, version),
                        verbose=False)
        parser = {
            'vizdata': VizData,
            'segmentationdata': SegmentationData,
        }[parser_name]
        traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                            grid_conf=grid_conf)
        valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                        grid_conf=grid_conf)

        trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                                shuffle=True,
                                                num_workers=nworkers,
                                                drop_last=True,
                                                worker_init_fn=worker_rnd_init)
        valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                                shuffle=False,
                                                num_workers=nworkers)

    return trainloader, valloader
