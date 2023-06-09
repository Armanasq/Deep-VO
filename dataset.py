import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from util import *

def load_dataset():
    path = "../data_odometry_gray/dataset/sequences/"
    pose_dir = "../data_odometry_poses/dataset/poses/"
    
    if mode == 'train':
        seq = [00, 1, 2, 4, 6, 8, 10]
    elif mode == 'val':
        seq = [4]
    elif mode == 'test':
        seq = [3,7,9]
        seq.extend(np.arange(11, 22))
    else:
        seq = None
        print("Please select the mode 'train', 'val', or 'test")
    
    for i in seq:
        if i <10:
            num = '0%s' %i
        else:
            num = '%s' % i
    seq_dir = path + num + '/'
    poses = pd.read_csv(pose_dir + ("%s.txt" % "04"), delimiter=' ', header=None)
    poses = np.array([np.array(poses.iloc[i]).reshape((3, 4)) for i in range(len(poses))])
    pose = poses[:,:3,3]
    rot = poses[:,:3,:3]
    
    quat = dcm2quat(rot)
    
    
    calib = pd.read_csv(seq_dir+'calib.txt', delimiter=' ', header=None, index_col=0)
    
    P0 = np.array(calib.loc['P0:']).reshape((3,4))
    P1 = np.array(calib.loc['P1:']).reshape((3,4))
    P2 = np.array(calib.loc['P2:']).reshape((3,4))
    P3 = np.array(calib.loc['P3:']).reshape((3,4))
    Tr = np.array(calib.loc['Tr:']).reshape((3,4))
    times = np.array(pd.read_csv(seq_dir + 'times.txt', delimiter=' ', header=None))
    
    image_left = os.listdir(seq_dir+'image_0')
    num_frames = len(image_left)
    bar = tqdm(total=num_frames, desc='Loading Images')
    images = []
    
    for i in range(num_frames):
        name = image_left[i]
        img = cv2.imread(seq_dir + 'image_0/' + name)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray_image)
        bar.update(1)
    bar.close
    
    return pose, quat, images
    
    