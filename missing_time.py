import os, glob, numpy as np, matplotlib.pyplot as plt, skimage, skimage.draw, scipy.io
import pandas as pd

#directory = 'C:/Users/Gordus_Lab/Documents/HsinYi/DeepLabCut_Anthony/8videos_1400frames_relabled/videos/aligned'
directory = 'B:/HsinYi/DeepLabCut_Anthony/8videos_1400frames_relabled/videos/aligned/newvideos'
os.makedirs(os.path.join(directory, '/croprot/'), exist_ok=True)
files = glob.glob(directory+'/*.npy')

missing_file_csv = 'B:/HsinYi/DeepLabCut_Anthony/8videos_1400frames_relabled/videos/aligned/newvideos/missing_time.csv'
test = pd.read_csv(missing_file_csv)

try:
    len(files) == len(test)
except:
    print("number of files do not match")

for i in range(len(files)):
    if files[i].split('\\')[1] == test.iloc[i][0].split('\\')[1]:
        joint_data = np.load(files[i])
        if np.isnan(test.iloc[i][1]):
            continue
        else:
            joint_new = joint_data[:,0:int(test.iloc[i][1])]
            np.save(files[i], joint_new)
    else:
        raise Exception("data file name does not match")
