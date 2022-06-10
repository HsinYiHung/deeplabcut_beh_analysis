## Content##



## Behavioral analysis pipeline

1. Use the DEEPLABCUT to track the side videos
2. Interpolation:
   * Run `check_model_labels.py` to do interpolation for joint coordinates. This helps remove bad estimation
3. Realign the videos:
    * Extract joint labeled from `_interpolation.npy` files
    * Run `croprot_videoalignment.npy`
      * Use joint coordinates to estimate the center and the orientation of the spider
      * Calculate the rotation matrix based on the first frame of the video
      * Rotate the videos to make them have the same orientation
4. Do interpolation again
  * Do interpolation on the `_croprotalignment.mp4`
5. Do joint angle analysis
6. Do behavioral analysis



## Python scripts
1. croprot.py: after labeling, we use the labeled coordinate to crop and rotate the images. And then, we used these new croprot images to retrain the network.
2. crop_post.py: after the DEEPLABCUT predicts joint coordinates for the video, we center the spider and crop the video to generate a new cropped video. Then, we ask the model to predict joint coordinate again.
3. croprot_post.py: after the DEEPLABCUT predicts joint coordinates for the video, we center the spider, crop, and rotate the video to generate a new cropped video. Then, we ask the model to predict joint coordinate again.
4. croprot_videoalignment.py: after labeling, we use the labeled coordinate to rotate the first image of the video. We apply this rotation matrix throughout the video to make every video align to each other.
