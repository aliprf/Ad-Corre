from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, ExpressionCodesAffectnet
from config import LearningConfig, InputDataSize, DatasetName, AffectnetConf, DatasetType

import numpy as np
import os
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import csv
from skimage.io import imread
import pickle
import csv
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from skimage import transform
from skimage.transform import resize
import tensorflow as tf
import random
import cv2
from skimage.feature import hog
from skimage import data, exposure
from matplotlib.path import Path
from scipy import ndimage, misc
from skimage.transform import SimilarityTransform, AffineTransform
from skimage.draw import rectangle
from skimage.draw import line, set_color


class CustomDataset:

    def create_dataset(self, img_filenames, anno_names, is_validation=False, ds=DatasetName.affectnet):
        def get_img(file_name):
            path = bytes.decode(file_name)
            image_raw = tf.io.read_file(path)
            img = tf.image.decode_image(image_raw, channels=3)
            img = tf.cast(img, tf.float32) / 255.0
            '''augmentation'''
            # if not (is_validation):# or tf.random.uniform([]) <= 0.5):
            #     img = self._do_augment(img)
            # ''''''
            return img

        def get_lbl(anno_name):
            path = bytes.decode(anno_name)
            lbl = load(path)
            return lbl

        def wrap_get_img(img_filename, anno_name):
            img = tf.numpy_function(get_img, [img_filename], [tf.float32])
            if is_validation and ds == DatasetName.affectnet:
                lbl = tf.numpy_function(get_lbl, [anno_name], [tf.string])
            else:
                lbl = tf.numpy_function(get_lbl, [anno_name], [tf.int64])

            return img, lbl

        epoch_size = len(img_filenames)

        img_filenames = tf.convert_to_tensor(img_filenames, dtype=tf.string)
        anno_names = tf.convert_to_tensor(anno_names)

        dataset = tf.data.Dataset.from_tensor_slices((img_filenames, anno_names))
        dataset = dataset.shuffle(epoch_size)
        dataset = dataset.map(wrap_get_img, num_parallel_calls=32) \
            .batch(LearningConfig.batch_size, drop_remainder=True) \
            .prefetch(10)
        return dataset
