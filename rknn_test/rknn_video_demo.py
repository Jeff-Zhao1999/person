from statistics import mode

import numpy as np
import cv2
from rknn.api import RKNN
from time import time
#import sys, getopt
#import sys
#sys.path.append(r'../src')

#from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
detection_model_path = './trained_models/detection_models/haarcascade_frontalface_default.xml'

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}

emotion_labels = get_labels('fer2013')

INPUT_SIZE = 64

def load_model(modle_path):
	# Create RKNN object
	rknn = RKNN()

	print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
	print('-->loading model')
	rknn.load_rknn(modle_path)
	print('loading model done')

	# init runtime environment
	print('--> Init runtime environment')
	ret = rknn.init_runtime()
	if ret != 0:
		print('Init runtime environment failed')
		exit(ret)
	print('done')
	print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
	return rknn

def model_predict(rknn, gray_face):
    emotion_prediction = rknn.inference(inputs=[gray_face], data_type='float32', data_format='nhwc')

    # perf
    #print('--> Begin evaluate model performance')
    #perf_results = rknn.eval_perf(inputs=[gray_face])
    #print(perf_results)
    #print('done')

    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]
    #print('emotion: ' + emotion_text)
    return emotion_text, emotion_probability

if __name__ == '__main__':
    # getting input model shapes for inference
    emotion_target_size = (INPUT_SIZE, INPUT_SIZE)
    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # starting lists for calculating modes
    emotion_window = []

    # loading models
    rknn = load_model('./fer2013_mini_XCEPTION.102-0.66.rknn')
    face_detection = load_detection_model(detection_model_path)

    # starting video streaming
    cv2.namedWindow('window_frame')
    video_capture = cv2.VideoCapture(0)
    while True:
        frame_start = time()
        bgr_image = video_capture.read()[1]
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)
        print('detect face cost time: %f'%(time() - frame_start))

        for face_coordinates in faces:
            start = time()
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_text, emotion_probability = model_predict(rknn, gray_face)
            print('predict emotion cost time: %f'%(time() - start))
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
               color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()
            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                      color, 0, 0, 1, 1)

        fps_text = ("%.2f" % (1 / (time() - frame_start)))
        cv2.putText(rgb_image, 'fps: ' +fps_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 1, cv2.LINE_AA)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
