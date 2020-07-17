First, ensure that the rknn-toolkit environment is normal

1. RKNN Conversion

python3.6 rknn_transfer.py

2. Predict a picture, and predicted picture(predicted_test_image_rknn.png) is created in current directory.

python3.6 rknn_image_demo.py test_images/1.jpg

3. Open the camera, predict video, if there is no camera you can modify the script to open the video file.

python3.6 rknn_video_demo.py
