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


class DataHelper:

    def create_generator_full_path(self, img_path, annotation_path, label=None):
        img_filenames, exp_filenames = self._create_image_and_labels_name_full_path(img_path=img_path,
                                                                                    annotation_path=annotation_path,
                                                                                    label=label)
        '''shuffle'''
        img_filenames, exp_filenames = shuffle(img_filenames, exp_filenames)
        return img_filenames, exp_filenames

    def _create_image_and_labels_name_full_path(self, img_path, annotation_path, label):
        img_filenames = []
        exp_filenames = []

        print('reading list -->')
        file_names = tqdm(os.listdir(img_path))
        print('<-')

        for file in file_names:
            if file.endswith(".jpg") or file.endswith(".png"):
                exp_lbl_file = str(file)[:-4] + "_exp.npy"  # just name

                if os.path.exists(annotation_path + exp_lbl_file):
                    if label is not None:
                        exp = np.load(annotation_path + exp_lbl_file)
                        if label is not None and exp != label:
                            continue

                    img_filenames.append(img_path + str(file))
                    exp_filenames.append(annotation_path + exp_lbl_file)

        return np.array(img_filenames), np.array(exp_filenames)

    def relabel_ds(self, labels):
        new_labels = np.copy(labels)

        index_src = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 31, 32, 36, 37, 38, 39, 40, 41, 48, 49, 50,
                     60, 61, 67, 59, 58]
        index_dst = [16, 15, 14, 13, 12, 11, 10, 9, 26, 25, 24, 23, 22, 35, 34, 45, 44, 43, 42, 47, 46, 54, 53, 52,
                     64, 63, 65, 55, 56]

        for i in range(len(index_src)):
            new_labels[index_src[i] * 2] = labels[index_dst[i] * 2]
            new_labels[index_src[i] * 2 + 1] = labels[index_dst[i] * 2 + 1]

            new_labels[index_dst[i] * 2] = labels[index_src[i] * 2]
            new_labels[index_dst[i] * 2 + 1] = labels[index_src[i] * 2 + 1]
        return new_labels