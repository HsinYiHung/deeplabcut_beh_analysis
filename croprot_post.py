#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 08:00:33 2022

@author: hsinyihung
"""
import h5py
import numpy as np, pandas as pd, math
import matplotlib.pyplot as plt
import os, glob
import cv2


### Read the HDF5 file
filename = "/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/011822  Spider Piezo 5Hz 75 182 With Pulses 2Sdelayed-01182022141158-0000-1DLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000.h5"
f1 = h5py.File(filename,'r+')
data_joints = f1['df_with_missing']['table'][:]

### Video name 
vid_name = '/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/011822  Spider Piezo 5Hz 75 182 With Pulses 2Sdelayed-01182022141158-0000-1.mp4'


### Read the pickle file
pickle_file = pd.read_pickle("/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/011822  Spider Piezo 5Hz 75 182 With Pulses 2Sdelayed-01182022141158-0000-1DLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000_meta.pickle")
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
        

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip

vid = imageio.get_reader(vid_name,  'ffmpeg')
import cv2
import numpy as np

cmap = matplotlib.cm.get_cmap('rainbow')

cap = cv2.VideoCapture(vid_name)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
buf_new = np.empty((frameCount, 400, 400, 3), np.dtype('uint8'))


fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    buf[fc]= cv2.cvtColor(buf[fc], cv2.COLOR_BGR2RGB)
    fc += 1

cap.release()




### Extract x y coordinate for anterior legs
anterior = joints[0:30,:]
posterior = joints[30:60,:]

anterior_x = anterior[::3]
anterior_y = anterior[1::3]
posterior_x = posterior[::3]
posterior_y = posterior[1::3]
prob_anterior = anterior[2::3]
prob_posterior = posterior[2::3]

anterior_x[np.where(prob_anterior<0.5)]= float('nan')
anterior_y[np.where(prob_anterior<0.5)]= float('nan')
posterior_x[np.where(prob_posterior<0.5)]= float('nan')
posterior_y[np.where(prob_posterior<0.5)]= float('nan')


for i in range(len(buf)):
    img = buf[i]
    h, w, color  = img.shape
        
    ax = np.nanmean(anterior_x,axis =0)[i]
    ay = np.nanmean(anterior_y,axis =0)[i]
        
    px = np.nanmean(posterior_x,axis =0)[i]
    py = np.nanmean(posterior_y,axis =0)[i]
        
        
    center_x = np.nanmean([ax,px])
    center_y = np.nanmean([ay,py])
    if np.isnan(center_x) or np.isnan(center_y):
        continue
    else: 
        y1 = int(center_y-200)
        y2 = int(center_y+200)
        x1 = int(center_x-200)
        x2 = int(center_x+200)
        if y1 >0 and y2<h and x1 >0 and x2<w:
        
            translation_y = center_y - 400/2
            translation_x = center_x - 400/2
        else:
            if y1<0 or y1==0:
                y1=0
                y2=400
            if y2>h or y2 ==h:
                y2=h
                y1=h-400
            if x1<0 or x1==0:
                x1=0
                x2=400
            if x2>w or x2==w:
                x2=w
                x1 = w-400
            translation_y = (y1+y2)/2 - 400/2
            translation_x = (x1+x2)/2 - 400/2
            
        
        
        new_img = img[y1:y2, x1:x2,:]
   
        slope = (py-ay)/(px-ax)
        n_h, n_w, n_color  = new_img.shape
        
        center = (n_h/2, n_w/2)
        
        if np.rad2deg(math.atan(slope))<0:
            if py<ay:
                rot_degree = (-90- np.rad2deg(math.atan(slope)))
            else:
                rot_degree = 180-(-90- np.rad2deg(math.atan(slope)))
                
        else:
            if py<ay:
                rot_degree = (90- np.rad2deg(math.atan(slope)))
            else:
                rot_degree = 180+(90- np.rad2deg(math.atan(slope)))
                
        M = cv2.getRotationMatrix2D(center, rot_degree, scale=1)
        rotated = cv2.warpAffine(new_img, M, (n_w,n_h))
        buf_new[i] = rotated

from cv2 import VideoWriter_fourcc
fourcc = VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(vid_name.replace(".mp4", "_croprot.mp4"),fourcc,100 , (400,400))

for i in range(len(buf_new)):
    out.write(buf_new[i])
out.release()


#FFMpegWriter = manimation.writers['ffmpeg']
#metadata = dict(title='Movie Test', artist='Matplotlib',
#                comment='Movie support!') 
#writer = FFMpegWriter(fps=20, metadata=metadata)        
#fig = plt.figure()
#im = plt.imshow(buf_new[0])        
#with writer.saving(fig, vid_name.replace(".mp4", "_croprot.mp4"), 300):
#    for i in range(frameCount):


#        im.set_data(buf_new[i])
        
#        writer.grab_frame()
        