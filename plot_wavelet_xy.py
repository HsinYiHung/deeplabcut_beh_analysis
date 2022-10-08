#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:41:07 2022

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
pwd = '/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/'
filename = '011822 Spider Prey2-01182022163621-0000-1_trimmedDLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000_interpolation.npy'
wavelet_file = '/Users/hsinyihung/Documents/PhD/JHU/Gordus lab/Spider_prey_vibration/behavioral_motifs_wavelet/8videos_1400frames_relabled/011822 Spider Prey2-01182022163621-0000-1_trimmed/wavelet/wavelet_dlc_euclidean_no-abspos_no-vel_11111000000000000000_50_zscore.npy'



joints = np.load(pwd+filename)
x = joints[::3]
y= joints[1::3]

position_temp = joints[:,1:joints.shape[1]]
position_temp = np.column_stack((position_temp, joints[:,0]))
position_diff = position_temp - joints


ampl = np.load(wavelet_file)


import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip
vid_name = pwd+filename
vid_name = vid_name.replace("_interpolation.npy", "_labeled.mp4")
vid = imageio.get_reader(vid_name,  'ffmpeg')

import cv2
import numpy as np

cmap = matplotlib.cm.get_cmap('rainbow')
rgb = []
rgb_angle=[]
num_joints = 20
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




#fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
#axs[0].hist(position_diff[12,:])
#axs[1].hist(position_diff[13,:])



#fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
#axs[0].plot(position_diff[12,:])
#axs[1].plot(position_diff[13,:])


TP_PER_LINE = int(np.ceil(ampl.shape[0] / 10))
nframes = ampl.shape[0]
xdata = np.arange(TP_PER_LINE)


fig = plt.figure()
##Plot Tibia-Femur joint
ax2 = plt.subplot(232)
lm2 = ax2.imshow(ampl[0:TP_PER_LINE, 0:int(ampl.shape[1] / 2)].T, interpolation='nearest', aspect='auto')
# ax2.xaxis.set_ticks([])
plt.title("Wavelet transform: x coordinates")
# plt.savefig(filename.replace(".h5", "_meta.png"), dpi = 3000)
ln2 = ax2.axvline(0, color='red')

ax3 = plt.subplot(233)
lm3 = ax3.imshow(ampl[0:TP_PER_LINE, int(ampl.shape[1] / 2):int(ampl.shape[1])].T, interpolation='nearest', aspect='auto')
# ax3.xaxis.set_ticks([])
plt.title("Wavelet transform: y coordinates")
ln3 = ax3.axvline(0, color='red')


ax4 = plt.subplot(235)

lm41, = ax4.plot(xdata,x[0,0:TP_PER_LINE])
lm42, = ax4.plot(xdata,x[1,0:TP_PER_LINE])
lm43, = ax4.plot(xdata,x[2,0:TP_PER_LINE])
lm44, = ax4.plot(xdata,x[3,0:TP_PER_LINE])
lm45, = ax4.plot(xdata,x[4,0:TP_PER_LINE])
ax4.set_ylim([0, int(x.max())])
# ax2.xaxis.set_ticks([])
plt.title("X coordinates")
ln4 = ax4.axvline(0, color='red')

ax5 = plt.subplot(236)

#ax5.legend(['me', 'ta', 'ti', 'pe', 'co'], loc='upper left')
#ax5.legend(['me', 'ta', 'ti', 'pe', 'co'], loc='lower right', bbox_to_anchor=(2, 0))

lm51, = ax5.plot(xdata, y[0,0:TP_PER_LINE])
lm52, = ax5.plot(xdata, y[1,0:TP_PER_LINE])
lm53, = ax5.plot(xdata, y[2,0:TP_PER_LINE])
lm54, = ax5.plot(xdata, y[3,0:TP_PER_LINE])
lm55, = ax5.plot(xdata, y[4,0:TP_PER_LINE])
ax5.set_ylim([0, int(y.max())])

plt.title("Y coordinates")
ln5 = ax5.axvline(0, color='red')

ax1 = plt.subplot(131)
im = ax1.imshow(buf[0])
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
fig.tight_layout()





fnameOut = wavelet_file.replace('.npy', '_xy.mp4')
with writer.saving(fig, fnameOut, 300):
    for i in range(frameCount):
        
        lm2.set_data(ampl[(0 + i):(TP_PER_LINE + i), 0:int(ampl.shape[1] / 2)].T)
        lm3.set_data(ampl[(0 + i):(TP_PER_LINE + i), int(ampl.shape[1] / 2):int(ampl.shape[1])].T)
        #ln4.set_xdata(i)
        #ln5.set_xdata(i)
        lm41.set_data(xdata, x[0,(0+i):(TP_PER_LINE+i)])
        lm42.set_data(xdata,x[1,(0+i):(TP_PER_LINE+i)])
        lm43.set_data(xdata,x[2,(0+i):(TP_PER_LINE+i)])
        lm44.set_data(xdata,x[3,(0+i):(TP_PER_LINE+i)])
        lm45.set_data(xdata,x[4,(0+i):(TP_PER_LINE+i)])
        lm51.set_data(xdata,y[0,(0+i):(TP_PER_LINE+i)])
        lm52.set_data(xdata,y[1,(0+i):(TP_PER_LINE+i)])
        lm53.set_data(xdata,y[2,(0+i):(TP_PER_LINE+i)])
        lm54.set_data(xdata,y[3,(0+i):(TP_PER_LINE+i)])
        lm55.set_data(xdata,y[4,(0+i):(TP_PER_LINE+i)])
        
        im.set_data(buf[i])

        writer.grab_frame()



