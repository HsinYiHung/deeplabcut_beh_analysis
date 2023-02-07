#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:37:00 2022

@author: hsinyihung
"""

def croprot_videoalignment(filename = None, joints=None):
    import h5py
    import numpy as np, pandas as pd, math
    import matplotlib.pyplot as plt
    import os, glob
    import cv2


    ### Read the HDF5 file
    #filename = "/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled/videos/1101 Spider Piezo 5Hz 0 107 With Pulses 2Sdelayed 2-11012021154002-0000-1DLC_resnet50_8videos_1400frames_relabledApr12shuffle1_50000.h5"

    #f1 = h5py.File(filename.split('/videos/')[0] +'/videos/aligned/'+filename.split('/videos/')[1],'r+')
    f1 = h5py.File(filename,'r+')
    data_joints = f1['df_with_missing']['table'][:]

    ### Video name 
    vid_name = filename.split('DLC')[0]+'.mp4'


    ### Read the pickle file
    pickle_file = pd.read_pickle(filename.replace('.h5', '_meta.pickle'))
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
#joints= np.load(filename.replace(".h5", "_interpolation.npy"))


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
    #buf_new = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))


    fc = 0
    ret = True

    while (fc < frameCount-1  and ret):
        try:
            ret, buf[fc] = cap.read()
            buf[fc]= cv2.cvtColor(buf[fc], cv2.COLOR_BGR2RGB)
            fc += 1
        except:
            pass
    cap.release()




    ### Extract x y coordinate for anterior legs
    anterior = np.copy(joints[0:30,:])
    posterior = np.copy(joints[30:60,:])

    anterior_x = np.copy(anterior[::3])
    anterior_y = np.copy(anterior[1::3])
    posterior_x = np.copy(posterior[::3])
    posterior_y = np.copy(posterior[1::3])
    prob_anterior = np.copy(anterior[2::3])
    prob_posterior = np.copy(posterior[2::3])

    anterior_x[np.where(prob_anterior<0.5)]= float('nan')
    anterior_y[np.where(prob_anterior<0.5)]= float('nan')
    posterior_x[np.where(prob_posterior<0.5)]= float('nan')
    posterior_y[np.where(prob_posterior<0.5)]= float('nan')


    i=0
    img = buf[i]
    h, w, color  = img.shape
    
    ax = np.nanmean(anterior_x,axis =0)[i]
    ay = np.nanmean(anterior_y,axis =0)[i]
    
    px = np.nanmean(posterior_x,axis =0)[i]
    py = np.nanmean(posterior_y,axis =0)[i]

    
    new_img = np.copy(img)
    slope = -(py-ay)/(px-ax)

    n_h, n_w, n_color  = new_img.shape
    
    center = (n_h/2, n_w/2)
    
    if np.rad2deg(math.atan(slope))>0:
        if py<ay:
            rot_degree = (90- np.rad2deg(math.atan(slope)))
        else:
            for i in range(len(buf)):
                buf[i] = cv2.flip(buf[i],0)
            rot_degree = (90- np.rad2deg(math.atan(slope)))
            #rot_degree = 180-(-90- np.rad2deg(math.atan(slope)))
            
    else:
        if py<ay:
            rot_degree = -(-90- np.rad2deg(math.atan(slope)))
        else:
            for i in range(len(buf)):
                buf[i] = cv2.flip(buf[i],0)
            rot_degree = -(-90- np.rad2deg(math.atan(slope)))
    if np.isnan(slope):
        rot_degree = 0
    M = cv2.getRotationMatrix2D(center, rot_degree, scale=1)
    
    joints_new = np.copy(joints)
    for j in range(0,60,3):
        tempx = joints[j]
        if py<ay:
            tempy = joints[j+1]
        else:
            tempy = n_h-joints[j+1]
        #newy = center[0]+M[0][0]*(tempy-center[0]) + (tempx-center[1])*M[1][0]
        #newx = center[1]+M[0][1]*(tempy-center[0]) + (tempx-center[1])*M[1][1]
        newy = M[1][1]*(tempy) + (tempx)*M[1][0]+M[1][2]
        newx = M[0][1]*(tempy) + (tempx)*M[0][0]+M[0][2]
        joints_new[j] = newx
        joints_new[j+1] = newy
        
            
    #data = f1['df_with_missing']
    #data_temp = np.copy(data['table'])
    
    cmap = matplotlib.cm.get_cmap('rainbow')
    rgb = []
    rgb_angle=[]
    
    
    for i in range(num_joints):
        rgb.append( list(cmap(i/num_joints)))
        
    
    
    for i in range(len(buf)):
        new_img = np.copy(buf[i])
        rotated = cv2.warpAffine(new_img, M, (n_w,n_h))
        #buf_new[i] = rotated
        buf[i] = rotated
        
        for j in range(0,60,3):
            c = tuple([s * 255 for s in rgb[int(j/3)]][0:3])
            c  =c[::-1]
            
            if joints_new[j+2][i]>0.5:
                
                #buf_new[i] = cv2.circle(buf_new[i], (int(joints_new[j][i]),int(joints_new[j+1][i])), radius=10, color=c, thickness=-1)
                buf[i] = cv2.circle(buf[i], (int(joints_new[j][i]), int(joints_new[j + 1][i])), radius=10,
                                        color=c, thickness=-1)
                buf[i] = cv2.putText(img=buf[i], text=str(i), fontScale=3.0,org = (200, 200),fontFace = cv2.FONT_HERSHEY_DUPLEX, color = (125, 246, 55))

        #f1['df_with_missing']['table'][:][i][1] = joints_new[:,i]
        #data['table'][:][i][1] = joints_new[:,i]
        #data_temp[:][i][1] = joints_new[:,i]
    #del data['table']
    #data['table'] = data_temp
    f1.close() 
    
    
    
    from cv2 import VideoWriter_fourcc
    fourcc = VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(vid_name.split('/videos')[0]+'/videos/aligned'+vid_name.split('/videos')[1].replace(".mp4", "_croprotaligned.mp4"),fourcc,100, (n_w, n_h) )
    #out = cv2.VideoWriter(
    #    vid_name.split('/videos')[0] + '/videos' + vid_name.split('/videos')[1].replace(".mp4",
    #                                                                                            "_croprotaligned.mp4"),
    #    fourcc, 100, (n_w, n_h))

    #for i in range(len(buf_new)):
    #    out.write(buf_new[i])
    #out.release()

    for i in range(len(buf)):
        out.write(buf[i])
    out.release()
    
    np.save(vid_name.split('/videos')[0]+'/videos/aligned'+vid_name.split('/videos')[1].replace(".mp4", "_croprotaligned.npy"), joints_new)
    #np.save(vid_name.split('/videos')[0] + '/videos' + vid_name.split('/videos')[1].replace(".mp4", "_croprotaligned.npy"),joints_new)
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
            