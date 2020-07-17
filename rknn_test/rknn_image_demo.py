import numpy as np
import cv2
from rknn.api import RKNN
from time import time
#from utils.builddata import preprocess_input
import sys, getopt
import sys

from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

# parameters for loading data and images
detection_model_path = './trained_models/detection_models/haarcascade_frontalface_default.xml'

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}

emotion_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_SIMPLEX

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

    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]
    return emotion_text


if __name__ == '__main__':
    image_path = sys.argv[1]

    # hyper-parameters for bounding boxes shape
    emotion_offsets = (0, 0)

    # loading models
    rknn = load_model('./fer2013_mini_XCEPTION.102-0.66.rknn')
    face_detection = load_detection_model(detection_model_path)

    # getting input model shapes for inference
    emotion_target_size = (INPUT_SIZE, INPUT_SIZE)

    # loading images
    rgb_image = load_image(image_path, grayscale=False)
    gray_image = load_image(image_path, grayscale=True)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    start = time()
    faces = detect_faces(face_detection, gray_image)
    print('detect face cost time: %f'%(time()-start))

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

        emotion_text = model_predict(rknn, gray_face)
        print('predict emotion cost time: %f'%(time()-start))
        print('predict emotion result: ' + emotion_text)

        color = (255, 0, 0)
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, 0, 1, 2)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./predicted_test_image_rknn.png', bgr_image)
