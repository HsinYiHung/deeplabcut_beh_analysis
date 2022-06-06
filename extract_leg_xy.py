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
filename = "/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/1101 Spider Piezo 5Hz 0 107 With Pulses 2Sdelayed 2-11012021154002-0000-1DLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000.h5"
f1 = h5py.File(filename,'r+')
data_joints = f1['df_with_missing']['table'][:]


### Read the pickle file
pickle_file = pd.read_pickle("/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/1101 Spider Piezo 5Hz 0 107 With Pulses 2Sdelayed 2-11012021154002-0000-1DLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000_meta.pickle")
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip
vid_name = filename.replace(".h5", "_labeled.mp4")
vid = imageio.get_reader(vid_name,  'ffmpeg')

import cv2
import numpy as np

cmap = matplotlib.cm.get_cmap('rainbow')
rgb = []
rgb_angle=[]

c=0
for i in range(num_joints):
    rgb.append( list(cmap(i/num_joints)))
    if (i % 5 ==3) or (i%5==4):
        continue
    else:
        rgb_angle.append(list(cmap(i/num_joints)))
        
        c+=1
        
cap = cv2.VideoCapture(vid_name)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    buf[fc]= cv2.cvtColor(buf[fc], cv2.COLOR_BGR2RGB)
    fc += 1

cap.release()

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!') 
writer = FFMpegWriter(fps=20, metadata=metadata)




fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(position_diff[12,:])
axs[1].hist(position_diff[13,:])



fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].plot(position_diff[12,:])
axs[1].plot(position_diff[13,:])


fig = plt.figure()
##Plot Tibia-Femur joint
 
ax4=plt.subplot(122)
lm4 = ax4.plot( np.arange(nframes)[4000:6000], position_diff[13,4000:6000])

#lm4 = ax4.plot( np.arange(nframes), joints[12,:])
#lm5 = ax4.plot( np.arange(nframes), joints[13,:])
ln4 = ax4.axvline(0, color='red')
plt.title("Left Anterior Tartus Y difference")
plt.legend()


ax1=plt.subplot(121)
im = ax1.imshow(buf[0])

fig.tight_layout()

with writer.saving(fig, filename.replace(".h5", "_leftanteriortartus_ydifference.mp4"), 100):
    for i in range(4000, 6000):

        ln4.set_xdata(i)

        im.set_data(buf[i])

        writer.grab_frame()



#fig = plt.figure()
##Plot Tibia-Femur joint
#ax2=plt.subplot(223)
#lm = ax2.plot( np.arange(nframes), joints[24,:])
#ax2.plot( np.arange(nframes), joints[25,:])

#ax2.xaxis.set_ticks([])
#plt.title("Right Anterior Metatarsus")
#plt.savefig(filename.replace(".h5", "_meta.png"), dpi = 3000)
#ln = ax2.axvline(0, color='red')

##Plot Metatarsus-Tibia joint
#fig = plt.figure()
#ax3=plt.subplot(224)
#lm3 = ax3.plot( np.arange(nframes), joints[27,:])
#ax3.plot( np.arange(nframes), joints[28,:])

#plt.title("Right Anterior Tarsus")
#plt.savefig(filename.replace(".h5", "_ti.png"), dpi = 3000)
#ln3 = ax3.axvline(0, color='red')

#ax4=plt.subplot(222)
#lm4 = ax4.plot( np.arange(nframes), joints[26,:], label='Right Anterior Metatarsus')
#ax4.plot( np.arange(nframes), joints[29,:], label='Right Anterior Tarsus')
#ln4 = ax4.axvline(0, color='red')
#plt.title("Probability")
#plt.legend()


#ax1=plt.subplot(221)
#im = ax1.imshow(buf[0])

#fig.tight_layout()

#with writer.saving(fig, filename.replace(".h5", "_rightanteriorleg_xy.mp4"), 100):
#    for i in range(frameCount):
#        ln.set_xdata(i)
#        ln3.set_xdata(i)
#        ln4.set_xdata(i)

#        im.set_data(buf[i])
        
#        writer.grab_frame()
        





