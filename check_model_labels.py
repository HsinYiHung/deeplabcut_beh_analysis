#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:26:02 2022

@author: hsinyihung
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:19:18 2022

@author: hsinyihung
"""


import h5py
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

### Read the HDF5 file
filename = "/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/101821 Spider Piezo 5Hz 0 107 With Pulses-10182021133907-0000-1DLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000.h5"
f1 = h5py.File(filename,'r+')
data_joints = f1['df_with_missing']['table'][:]



### Read the pickle file
pickle_file = pd.read_pickle("/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/101821 Spider Piezo 5Hz 0 107 With Pulses-10182021133907-0000-1DLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000_meta.pickle")
pickle_file = pickle_file['data']
fps = pickle_file['fps']
nframes = pickle_file['nframes']
num_joints = pickle_file['DLC-model-config file']['num_joints']
all_joints_names = pickle_file['DLC-model-config file']['all_joints_names']

for i in range (nframes):
    if i==0:
        joints = data_joints[i][1]
    else:
        joints = np.column_stack((joints,data_joints[i][1]))

position_temp = joints[:,1:joints.shape[1]]
position_temp = np.column_stack((position_temp, joints[:,0]))
position_diff = position_temp - joints


### Plot the difference distribution
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].plot(joints[12,:])
axs[1].plot(joints[13,:])

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].plot(position_diff[12,:])
axs[1].plot(position_diff[13,:])
#axs[0].set_ylim([-200,200])

index = np.where((np.abs(position_diff)>50))
for i in range (len(index[0])):
    row = index[0][i]
    column = index[1][i]
    if column >joints.shape[1]-3:
        continue
    elif np.abs(joints[row, column]-joints[row, column+1])>50:
        if np.abs(joints[row, column]-joints[row, column+2])<50:
            joints[row, column+1] = (joints[row, column]+joints[row, column+2])/2

position_temp = joints[:,1:joints.shape[1]]
position_temp = np.column_stack((position_temp, joints[:,0]))
position_diff = position_temp - joints


np.save(filename.replace(".h5", "_interpolation.npy"), joints)





