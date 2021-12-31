import tensorflow as tf
import os
import pathlib

import numpy as np
import zipfile

import matplotlib.pyplot as plt
from PIL import Image

import cv2 

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import streamlit as st

# 내 로컬에 설치된 레이블 파일을, 인덱스와 연결시킨다.
PATH_TO_LABELS = 'C:\\Users\\a\\Desktop\\yh_study\\TensorFlow\\models\research\\object_detection\\data\\mscoco_label_map.pbtxt'

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

print(category_index)

# 모델 로드하는 함수.

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
# 위의 사이트에서 모델을 가져올수있다.

# /20200711/efficientdet_d0_coco17_tpu-32.tar.gz
# /20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz

# Download and extract model
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

def load_model(model_dir) :
    model_full_dir = model_dir + "/saved_model"

    # Load saved model and build the detection function
    detection_model = tf.saved_model.load(model_full_dir)
    return detection_model

detection_model = load_model(PATH_TO_MODEL_DIR)
# print()
# print(detection_model.signatures['serving_default'].inputs)
# print()
# print(detection_model.signatures['serving_default'].output_dtypes)
# print()
# print(detection_model.signatures['serving_default'].output_shapes)


def show_inference(detection_model, image_np) :
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detection_model(input_tensor)
#     print(detections)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    # print(detections)
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          agnostic_mode=False)

    img = Image.fromarray(image_np_with_detections)
    st.image(img,use_column_width=True)


def run_object_detection(image_np) :
    show_inference(detection_model, image_np)