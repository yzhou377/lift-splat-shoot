"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

# This is the top-level script that you can run python with 
import src
from fire import Fire
# Fire is a google library for executing CML commands inside a python script. 
# To run this work with evaluation mode:
# python main.py eval_model_iou mini --modelf=/home/m/lift-splat-shoot/model_weights/model525000.pt --dataroot=/home/m/Downloads
# where version=mini, and the other two argument follows. Remember to specify the GPUID as well 

if __name__ == '__main__':
    Fire({
        'lidar_check': src.explore.lidar_check,
        'cumsum_check': src.explore.cumsum_check,

        'train': src.train.train,
        'eval_model_iou': src.explore.eval_model_iou,
        'viz_model_preds': src.explore.viz_model_preds,
    })