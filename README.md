
Python scripts:
1. croprot.py: after labeling, we use the labeled coordinate to crop and rotate the images. And then, we used these new croprot images to retrain the network. 
2. crop_post.py: after the DEEPLABCUT predicts joint coordinates for the video, we center the spider and crop the video to generate a new cropped video. Then, we ask the model to predict joint coordinate again. 
3. croprot_post.py: after the DEEPLABCUT predicts joint coordinates for the video, we center the spider, crop, and rotate the video to generate a new cropped video. Then, we ask the model to predict joint coordinate again. 
4. croprot_videoalignment.py: after labeling, we use the labeled coordinate to crop and rotate the images. And then, we used these new croprot images to retrain the network. 