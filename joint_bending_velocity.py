#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:47:05 2022

@author: hsinyihung
"""

t1= 200
t2 = 250

import h5py
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

### Read the HDF5 file
filename = "/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/012122 Spider Piezo 5Hz 75 182 With Pulses 2Sdelayed 2-01212022132058-0000-1_trimmedDLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000.h5"
f1 = h5py.File(filename,'r+')
data_joints = f1['df_with_missing']['table'][:]


### Read the pickle file
pickle_file = pd.read_pickle("/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/012122 Spider Piezo 5Hz 75 182 With Pulses 2Sdelayed 2-01212022132058-0000-1_trimmedDLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000_meta.pickle")
pickle_file = pickle_file['data']
fps = pickle_file['fps']
nframes = pickle_file['nframes']
num_joints = pickle_file['DLC-model-config file']['num_joints']
all_joints_names = pickle_file['DLC-model-config file']['all_joints_names']

#for i in range (nframes):
#    if i==0:
#        joints = data_joints[i][1]
#    else:
#        joints = np.column_stack((joints,data_joints[i][1]))

### Load interpolated data
joints= np.load(filename.replace(".h5", "_interpolation.npy"))


position_temp = joints[:,1:joints.shape[1]]
position_temp = np.column_stack((position_temp, joints[:,0]))
position_diff = position_temp - joints


index = np.where((np.abs(position_diff)>50))
for i in range (len(index[0])):
    row = index[0][i]
    column = index[1][i]
    if column >joints.shape[1]-3:
        continue
    elif np.abs(joints[row, column]-joints[row, column+1])>50:
        if np.abs(joints[row, column]-joints[row, column+2])<50:
            joints[row, column+1] = (joints[row, column]+joints[row, column+2])/2

position_temp = joints[3:num_joints*3]
position_temp = np.vstack((position_temp, joints[0:3]))
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
    


print("fps:", fps)

fps =100

radian_temp= radian[:,1:radian.shape[1]]

radian_diff = radian_temp - radian[:, 0:radian.shape[1]-1]

joint_bending_v = joint_bending_v = radian_diff/fps*1000




fig = plt.figure()
##Plot Tibia-Femur joint
ax2=plt.subplot(321)
#lm = ax2.plot( np.arange(nframes)[t1:t2], joint_bending_v[0, t1:t2])
ax2.plot( np.arange(nframes)[t1:t2], joint_bending_v[3, t1:t2])
ln = ax2.axvline(t1, color='red')
ax2.xaxis.set_ticks([])
plt.title("Tibia-Femur joint")
#plt.savefig(filename.replace(".h5", "_meta.png"), dpi = 3000)


##Plot Metatarsus-Tibia joint
#fig = plt.figure()
ax3=plt.subplot(323)
#lm3 = ax3.plot( np.arange(nframes)[t1:t2], joint_bending_v[1, t1:t2])
ax3.plot( np.arange(nframes)[t1:t2], joint_bending_v[4, t1:t2])
ln3 = ax3.axvline(t1, color='red')
plt.title('Metatarsus-Tibia joint')
#plt.savefig(filename.replace(".h5", "_ti.png"), dpi = 3000)


#fig = plt.figure()
ax4=plt.subplot(325)
#m4 = ax4.plot( np.arange(nframes)[t1:t2], joint_bending_v[2, t1:t2])
ax4.plot( np.arange(nframes)[t1:t2], joint_bending_v[5, t1:t2])
ln4 = ax4.axvline(t1, color='red')
plt.title('Metatarsal joint')
#plt.savefig(filename.replace(".h5", "_fe.png"), dpi = 3000)

ax1=plt.subplot(122)
im = ax1.imshow(np.rot90(buf[0, :, :, :]))
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)   
fig.tight_layout()

with writer.saving(fig, filename.replace(".h5", "_joint_bending_v.mp4"), 300):
    for i in range(t1,t2):
        ln.set_xdata(i)
        ln3.set_xdata(i)
        ln4.set_xdata(i)

        im.set_data(np.rot90(buf[i]))
        
        writer.grab_frame()
        

fig = plt.figure()
##Plot Tibia-Femur joint
ax2=plt.subplot(321)
#lm = ax2.plot( np.arange(nframes)[t1:t2], radian[0, t1:t2])
ax2.plot( np.arange(nframes)[t1:t2], radian[3, t1:t2])
ln = ax2.axvline(t1, color='red')
ax2.xaxis.set_ticks([])
plt.title("Tibia-Femur joint")
#plt.savefig(filename.replace(".h5", "_meta.png"), dpi = 3000)


##Plot Metatarsus-Tibia joint
#fig = plt.figure()
ax3=plt.subplot(323)
#lm3 = ax3.plot( np.arange(nframes)[t1:t2], radian[1, t1:t2])
ax3.plot( np.arange(nframes)[t1:t2], radian[4, t1:t2])
ln3 = ax3.axvline(t1, color='red')
plt.title('Metatarsus-Tibia joint')
#plt.savefig(filename.replace(".h5", "_ti.png"), dpi = 3000)


#fig = plt.figure()
ax4=plt.subplot(325)
#lm4 = ax4.plot( np.arange(nframes)[t1:t2], radian[2, t1:t2])
ax4.plot( np.arange(nframes)[t1:t2], radian[5, t1:t2])
ln4 = ax4.axvline(t1, color='red')
plt.title('Metatarsal joint')
#plt.savefig(filename.replace(".h5", "_fe.png"), dpi = 3000)

ax1=plt.subplot(122)
im = ax1.imshow(np.rot90(buf[0, :, :, :]))
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)   
fig.tight_layout()




span_left_tarsus = np.sqrt(np.square(joints[12, :]-joints[42, :])+np.square(joints[13, :]-joints[43, :]))
span_right_tarsus = np.sqrt(np.square(joints[27, :]-joints[57, :])+np.square(joints[28, :]-joints[58, :]))
fig = plt.figure()
##Plot Tibia-Femur joint
ax2=plt.subplot(211)
lm = ax2.plot( np.arange(nframes)[t1:t2], span_left_tarsus[t1:t2])
#ln = ax2.axvline(t1, color='red')
ax2.xaxis.set_ticks([])
plt.title("Left tarsus leg span")
#plt.savefig(filename.replace(".h5", "_meta.png"), dpi = 3000)


##Plot Metatarsus-Tibia joint
#fig = plt.figure()
ax3=plt.subplot(212)
lm3 = ax3.plot( np.arange(nframes)[t1:t2], span_right_tarsus[t1:t2])
#ax3.plot( np.arange(nframes)[t1:t2], radian[4, t1:t2])
#ln3 = ax3.axvline(t1, color='red')
plt.title('Right tarsus leg span')
#plt.savefig(filename.replace(".h5", "_ti.png"), dpi = 3000)