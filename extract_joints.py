#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:23:52 2022

@author: hsinyihung
"""

import h5py
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

### Read the HDF5 file
filename = "/Users/hsinyihung/Documents/DeepLabCut/80_frames-HY-2022-02-14/videos/0906 Side Spider Prey-09062021141012-0000-1_trimmedDLC_resnet50_0906 Spider Annotating 80 framesNov13shuffle1_50000.h5"
f1 = h5py.File(filename,'r+')
data_joints = f1['df_with_missing']['table'][:]


### Read the pickle file
pickle_file = pd.read_pickle("/Users/hsinyihung/Documents/DeepLabCut/80_frames-HY-2022-02-14/videos/0906 Side Spider Prey-09062021141012-0000-1_trimmedDLC_resnet50_0906 Spider Annotating 80 framesNov13shuffle1_50000_meta.pickle")
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
        
position_temp = joints[3:num_joints*3]
position_temp = np.vstack((position_temp, joints[0:3]))
position_diff = position_temp - joints


for i in range (num_joints*3):
    if i % 3 ==2:
        position_diff[i]= (position_temp[i]+joints[i])/2
    if i % 15 == 14:
        position_diff[i-2:i+1]= 0
        

        
for i in range (0, num_joints*3-3,3):

    inner = position_diff[i]*position_diff[i+3]+position_diff[i+1]*position_diff[i+4]
    norms = np.sqrt(position_diff[i]**2+position_diff[i+1]**2)*np.sqrt(position_diff[i+3]**2+position_diff[i+4]**2)
    probs = (position_diff[i+2]+position_diff[i+5])/2
    if len(np.argwhere(norms ==0)) == nframes:
        continue
    else:
        cos = inner / norms
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        deg = np.rad2deg(rad)
    
        if i==0:
            cosine = cos
            radian = rad
            degree = deg
            probability = probs
        else:
            cosine = np.vstack((cosine, cos))
            radian = np.vstack((radian, rad))
            degree = np.vstack((degree, deg))
            probability = np.vstack((probability, probs))

degree = 180-degree
degree = np.abs(degree)

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

for i in range(len(rgb_angle)):
    rgb_angle[i] = np.reshape(rgb_angle[i]*nframes, (nframes,4))
    
    rgb_angle[i][:,3] = probability[i]
    

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




fig = plt.figure()
##Plot Tibia-Femur joint
ax2=plt.subplot(222)
lm = ax2.scatter( np.arange(nframes), degree[0], s=0.5, color = rgb_angle[0])
ax2.scatter( np.arange(nframes), degree[3], s=0.5, color = rgb_angle[3])
ax2.scatter( np.arange(nframes), degree[6], s=0.5, color = rgb_angle[6])
ax2.scatter( np.arange(nframes), degree[9], s=0.5, color = rgb_angle[9])
ax2.xaxis.set_ticks([])
plt.title("Tibia-Femur joint")
#plt.savefig(filename.replace(".h5", "_meta.png"), dpi = 3000)
ln = ax2.axvline(0, color='red')

##Plot Metatarsus-Tibia joint
#fig = plt.figure()
ax3=plt.subplot(223)
lm3 = ax3.scatter( np.arange(nframes), degree[1], s=0.5, color = rgb_angle[1])
ax3.scatter( np.arange(nframes), degree[4], s=0.5, color = rgb_angle[4])
ax3.scatter( np.arange(nframes), degree[7], s=0.5, color = rgb_angle[7])
ax3.scatter( np.arange(nframes), degree[10], s=0.5, color = rgb_angle[10])
plt.title('Metatarsus-Tibia joint')
#plt.savefig(filename.replace(".h5", "_ti.png"), dpi = 3000)
ln3 = ax3.axvline(0, color='red')

#fig = plt.figure()
ax4=plt.subplot(224)
lm4 = ax4.scatter( np.arange(nframes), degree[2], s=0.5, color = rgb_angle[2])
ax4.scatter( np.arange(nframes), degree[5], s=0.5, color = rgb_angle[5])
ax4.scatter( np.arange(nframes), degree[8], s=0.5, color = rgb_angle[8])
ax4.scatter( np.arange(nframes), degree[11], s=0.5, color = rgb_angle[11])
plt.title('Metatarsal joint')
#plt.savefig(filename.replace(".h5", "_fe.png"), dpi = 3000)
ln4 = ax4.axvline(0, color='red')

ax1=plt.subplot(221)
im = ax1.imshow(buf[0])
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
fig.tight_layout()

with writer.saving(fig, filename.replace(".h5", "_joint_angle.mp4"), 100):
    for i in range(frameCount):
        ln.set_xdata(i)
        ln3.set_xdata(i)
        ln4.set_xdata(i)

        im.set_data(buf[i])
        
        writer.grab_frame()
        
        
        
t1=0
t2=500


fig = plt.figure()
##Plot Tibia-Femur joint
ax2=plt.subplot(321)
lm = ax2.plot( np.arange(nframes)[t1:t2], degree[0, t1:t2])
ax2.plot( np.arange(nframes)[t1:t2], degree[3, t1:t2])
ln = ax2.axvline(t1, color='red')
ax2.xaxis.set_ticks([])
plt.title("Tibia-Femur joint")
#plt.savefig(filename.replace(".h5", "_meta.png"), dpi = 3000)


##Plot Metatarsus-Tibia joint
#fig = plt.figure()
ax3=plt.subplot(323)
lm3 = ax3.plot( np.arange(nframes)[t1:t2], degree[1, t1:t2])
ax3.plot( np.arange(nframes)[t1:t2], degree[4, t1:t2])
ln3 = ax3.axvline(t1, color='red')
plt.title('Metatarsus-Tibia joint')
#plt.savefig(filename.replace(".h5", "_ti.png"), dpi = 3000)


#fig = plt.figure()
ax4=plt.subplot(325)
lm4 = ax4.plot( np.arange(nframes)[t1:t2], degree[2, t1:t2])
ax4.plot( np.arange(nframes)[t1:t2], degree[5, t1:t2])
ln4 = ax4.axvline(t1, color='red')
plt.title('Metatarsal joint')
#plt.savefig(filename.replace(".h5", "_fe.png"), dpi = 3000)

ax1=plt.subplot(122)
im = ax1.imshow(np.rot90(buf[0, :, :, :]))
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)   
fig.tight_layout()

with writer.saving(fig, filename.replace(".h5", "_joint_angle.mp4"), 300):
    for i in range(t1,t2):
        ln.set_xdata(i)
        ln3.set_xdata(i)
        ln4.set_xdata(i)

        im.set_data(np.rot90(buf[i]))
        
        writer.grab_frame()
        