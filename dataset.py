import numpy as np
import pandas as pd


def load_dataset():
    path = "../data_odometry_gray/dataset/sequences/"
    pose_dir = "../data_odometry_poses/dataset/poses/"
    
    if mode == 'train':
        seq = [00, 1, 2, 4, 6, 8, 10]
    elif mode == 'val':
        seq = [5]
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
    seq_dir = path + num
    pose = pd.read_csv(pose_dir+num+'.txt', delimiter=' ', header=None)
    calib = pd.read_csv(seq_dir+'calib.txt', delimiter=' ', header=None, index_col=0)
    
    P0 = np.array(calib.loc['P0:']).reshape((3,4))
    P1 = np.array(calib.loc['P1:']).reshape((3,4))
    P2 = np.array(calib.loc['P2:']).reshape((3,4))
    P3 = np.array(calib.loc['P3:']).reshape((3,4))
    Tr = np.array(calib.loc['Tr:']).reshape((3,4))
    times = np.array(pd.read_csv(seq_dir + 'times.txt', delimiter=' ', header=None))
    
    
    num_frames = len(left_images)
        
        

i = 9

    
print(num)