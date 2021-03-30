"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import torchvision 
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info, tensorboard_visualiza


#YZ IMPORT 
import matplotlib.pyplot as plt 
from PIL import Image  
#from .data_carla import compile_data_CARLA

def train(version,
            dataroot='/data/nuscenes',
            nepochs=10000,
            gpuid=0,

            H=900, W=1600,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4), 
            rand_flip=True,
            ncams=6, # choose to drop one camera out of the 6 cameras 
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',

            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=5,
            nworkers=10,
            lr=1e-3,
            weight_decay=1e-7,
            ):

    # Differentiating between the CARLA trainer and the Nuscenes trainer here. 
    if version=="CARLA":
        grid_conf = {
            'xbound': xbound,
            'ybound': ybound,
            'zbound': zbound,
            'dbound': dbound,
        }
        data_aug_conf = {
                        'resize_lim': resize_lim,
                        'final_dim': final_dim,
                        'rot_lim': rot_lim,
                        'H': H, 'W': W,
                        'rand_flip': rand_flip,
                        'bot_pct_lim': bot_pct_lim,
                        'cams': ['front_left', 'front', 'front_right',
                                'rear_left', 'rear', 'rear_right'],
                        'Ncams': ncams,
        }   
        # Here is the CARLA data compilation part
        trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                            grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                            parser_name='segmentationdata')

                
    else:
        grid_conf = {
            'xbound': xbound,
            'ybound': ybound,
            'zbound': zbound,
            'dbound': dbound,
        }
        data_aug_conf = {
                        'resize_lim': resize_lim,
                        'final_dim': final_dim,
                        'rot_lim': rot_lim,
                        'H': H, 'W': W,
                        'rand_flip': rand_flip,
                        'bot_pct_lim': bot_pct_lim,
                        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                        'Ncams': ncams,
        }

        # Here is the NuScene data compilation part
        trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                            grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                            parser_name='segmentationdata')


    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    
    # LOADING CHECK POINT
    model.load_state_dict(torch.load("/home/m/lift-splat-shoot/runs/model130000.pt"))
    
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 1000 if version == 'mini' else 10000

    model.train()
    counter = 0
    for epoch in range(nepochs):
        print("Epoch: %10d" % epoch)
        np.random.seed()
        for batchi, (orig_imgs, imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            '''
            batchi is the batch id
            imgs are normalized tensor form images of the size: 
                [nbatch,ncam,n_channel(3 for RGB), img_height, img_width]
            rots are rotation of the sensor w.r.t. the vehicle center
            trans are translation 
            '''
            # binimgs is the BEV view map (20k0*200)
            # binimgs.shape= torch.Size([4, 1, 200, 200])
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            binimgs = binimgs.to(device)    
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)
                # Adding the tensorboard image display
                sampeld_image_data= iter(trainloader)
                tensorboard_visualiza(model= model, writer= writer, dataloader= trainloader, is_train=1, device= device)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
