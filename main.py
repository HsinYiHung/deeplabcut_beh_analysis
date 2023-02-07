#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:24:39 2022

@author: hsinyihung
"""
from check_model_labels import *
from croprot_videoalignment import *
import os, glob


#directory = 'C:/Users/Hsin-Yi/Documents/GitHub/DeepLabCut/DeepLabCut_Anthony/videos'
directory ='B:/HsinYi/DeepLabCut_Anthony/8videos_1400frames_relabled/videos'

files = glob.glob(directory+'/*DLC*.h5')

for i in range(len(files)):
    filename = files[i]
    joints = check_model_labels(filename)
    croprot_videoalignment(filename, joints)


