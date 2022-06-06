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


filename = "/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled_croprot/labeled-data/111021 Spider Prey-11102021162153-0000-1/CollectedData_hy.csv"
df2 = pd.read_csv(filename, header = None)

main_dir = '/Users/hsinyihung/Documents/DeepLabCut/8videos_1400frames_relabled_croprot'



### Extract x y coordinate for anterior legs
anterior = df2.iloc[:, 1:21]
posterior = df2.iloc[:, 21:41]

anterior_x = anterior[anterior.columns[::2]].drop([0,1,2]).astype(float)
anterior_y = anterior[anterior.columns[1::2]].drop([0,1,2]).astype(float)
posterior_x = posterior[posterior.columns[::2]].drop([0,1,2]).astype(float)
posterior_y = posterior[posterior.columns[1::2]].drop([0,1,2]).astype(float)


file = df2[0]
file = list(df2[0])
files  = file[3:len(file)]

if len(files) == len(anterior_x):
    for i in range(len(files)):
        file = main_dir + '/'+files[i]
        img = cv2.imread(file)
        h, w, color  = img.shape
        
        ax = anterior_x.mean(axis =1).loc[i+3]
        ay = anterior_y.mean(axis =1).loc[i+3]
        
        px = posterior_x.mean(axis =1).loc[i+3]
        py = posterior_y.mean(axis =1).loc[i+3]
        
        
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
            #new_img = img[int(center_y-200):int(center_y+200), int(center_x-200):int(center_x+200),:]
            slope = (py-ay)/(px-ax)
            n_h, n_w, n_color  = new_img.shape
            center = (n_h/2, n_w/2)
            if np.rad2deg(math.atan(slope))<0:
                if ay<py:
                    rot_degree = (0- np.rad2deg(math.atan(slope)))
                else:
                    rot_degree = (0- np.rad2deg(math.atan(slope)))
                    rot_degree = rot_degree+180
            else:
                if ay<py:
                    rot_degree = (180- np.rad2deg(math.atan(slope)))
                else:
                    rot_degree = (360- np.rad2deg(math.atan(slope)))
                
            M = cv2.getRotationMatrix2D(center, rot_degree, scale=1)
            rotated = cv2.warpAffine(new_img, M, (n_w,n_h))
            cv2.imwrite(file, rotated)
            
            
            
            
            
            
            for j in range(1,40,2):
                tempx = float(df2.loc[i+3][j])- translation_x
                tempy = float(df2.loc[i+3][j+1])- translation_y
                newy = center[0]+M[0][0]*(tempy-center[0]) + (tempx-center[1])*M[1][0]
                newx = center[1]+M[0][1]*(tempy-center[0]) + (tempx-center[1])*M[1][1]
                df2.loc[i+3][j] = str(newx)
                df2.loc[i+3][j+1] = str(newy)
    df2.to_csv(filename, index =False, header=False)

#fig, ax = plt.subplots()
#fig = plt.imshow(img)
#ax.plot((float(df2.loc[i+2][j])), (float(df2.loc[i+2][j+1])), 'o', color='y')                

else:
    print("files number not match")



### Extract x y coordinate for four legs
#anterior_left = df2.iloc[:, 1:11]
#anterior_left_x =anterior_left[anterior_left.columns[::2]]
#anterior_left_y =anterior_left[anterior_left.columns[1::2]]

#anterior_right = df2.iloc[:, 11:21]
#anterior_right_x =anterior_right[anterior_right.columns[::2]]
#anterior_right_y =anterior_right[anterior_right.columns[1::2]]

#posterior_left = df2.iloc[:, 21:31]
#posterior_left_x =posterior_left[posterior_left.columns[::2]]
#posterior_left_y =posterior_left[posterior_left.columns[1::2]]

#posterior_right = df2.iloc[:, 31:41]
#posterior_right_x =posterior_right[posterior_right.columns[::2]]
#posterior_right_y =posterior_right[posterior_right.columns[1::2]]


### Convert dataframe to float
#anterior_left_x = anterior_left_x.drop([0,1]).astype(float)
#anterior_left_y = anterior_left_y.drop([0,1]).astype(float)
#anterior_right_x = anterior_right_x.drop([0,1]).astype(float)
#anterior_right_y = anterior_right_y.drop([0,1]).astype(float)

#posterior_left_x = posterior_left_x.drop([0,1]).astype(float)
#posterior_left_y = posterior_left_y.drop([0,1]).astype(float)
#posterior_right_x = posterior_right_x.drop([0,1]).astype(float)
#posterior_right_y = posterior_right_y.drop([0,1]).astype(float)