from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, ExpressionCodesAffectnet
from config import LearningConfig, InputDataSize, DatasetName, AffectnetConf, DatasetType

import numpy as np
import os
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray, savez_compressed, savez
import csv
from skimage.io import imread
import pickle
import csv
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
import random
import cv2
from skimage.feature import hog
from skimage import data, exposure
from matplotlib.path import Path
from scipy import ndimage, misc
from data_helper import DataHelper
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from dataset_class import CustomDataset
from sklearn.metrics import precision_recall_fscore_support as score


class AffectNet:
    def __init__(self, ds_type):
        """we set the parameters needed during the whole class:
        """
        self.ds_type = ds_type
        if ds_type == DatasetType.train:
            self.img_path = AffectnetConf.no_aug_train_img_path
            self.anno_path = AffectnetConf.no_aug_train_annotation_path
            self.img_path_aug = AffectnetConf.aug_train_img_path
            self.masked_img_path = AffectnetConf.aug_train_masked_img_path
            self.anno_path_aug = AffectnetConf.aug_train_annotation_path

        elif ds_type == DatasetType.eval:
            self.img_path_aug = AffectnetConf.eval_img_path
            self.anno_path_aug = AffectnetConf.eval_annotation_path
            self.img_path = AffectnetConf.eval_img_path
            self.anno_path = AffectnetConf.eval_annotation_path
            self.masked_img_path = AffectnetConf.eval_masked_img_path

        elif ds_type == DatasetType.train_7:
            self.img_path = AffectnetConf.no_aug_train_img_path_7
            self.anno_path = AffectnetConf.no_aug_train_annotation_path_7
            self.img_path_aug = AffectnetConf.aug_train_img_path_7
            self.masked_img_path = AffectnetConf.aug_train_masked_img_path_7
            self.anno_path_aug = AffectnetConf.aug_train_annotation_path_7

        elif ds_type == DatasetType.eval_7:
            self.img_path_aug = AffectnetConf.eval_img_path_7
            self.anno_path_aug = AffectnetConf.eval_annotation_path_7
            self.img_path = AffectnetConf.eval_img_path_7
            self.anno_path = AffectnetConf.eval_annotation_path_7
            self.masked_img_path = AffectnetConf.eval_masked_img_path_7

    def test_accuracy(self, model, print_samples=False):
        dhp = DataHelper()

        batch_size = LearningConfig.batch_size
        exp_pr_glob = []
        exp_gt_glob = []
        acc_per_label = []
        '''create batches'''
        img_filenames, exp_filenames = dhp.create_generator_full_path(
            img_path=self.img_path,
            annotation_path=self.anno_path, label=None)

        print(len(img_filenames))
        step_per_epoch = int(len(img_filenames) // batch_size)
        exp_pr_lbl = []
        exp_gt_lbl = []

        cds = CustomDataset()
        ds = cds.create_dataset(img_filenames=img_filenames,
                                anno_names=exp_filenames,
                                is_validation=True, ds=DatasetName.affectnet)
        batch_index = 0
        for img_batch, exp_gt_b in ds:
            '''predict on batch'''
            exp_gt_b = exp_gt_b[:, -1]
            img_batch = img_batch[:, -1, :, :]

            # probab_exp_pr_b, _ = model.predict_on_batch([img_batch])  # with embedding
            pr_data = model.predict_on_batch([img_batch])
            probab_exp_pr_b = pr_data[0]

            scores_b = np.array([tf.nn.softmax(probab_exp_pr_b[i]) for i in range(len(probab_exp_pr_b))])
            exp_pr_b = np.array([np.argmax(scores_b[i]) for i in range(len(probab_exp_pr_b))])

            if print_samples:
                for i in range(len(exp_pr_b)):
                    dhp.test_image_print_exp(str(i) + str(batch_index + 1), np.array(img_batch[i]),
                                             np.int8(exp_gt_b[i]), np.int8(exp_pr_b[i]))

            exp_pr_lbl += np.array(exp_pr_b).tolist()
            exp_gt_lbl += np.array(exp_gt_b).tolist()
            batch_index += 1
        exp_pr_lbl = np.int64(np.array(exp_pr_lbl))
        exp_gt_lbl = np.int64(np.array(exp_gt_lbl))

        global_accuracy = accuracy_score(exp_gt_lbl, exp_pr_lbl)

        precision, recall, fscore, support = score(exp_gt_lbl, exp_pr_lbl)

        conf_mat = confusion_matrix(exp_gt_lbl, exp_pr_lbl) / 500.0
        # conf_mat = tf.math.confusion_matrix(exp_gt_lbl, exp_pr_lbl, num_classes=7)/500.0
        avg_acc = np.mean([conf_mat[i, i] for i in range(7)])

        ds = None
        face_img_filenames = None
        eyes_img_filenames = None
        nose_img_filenames = None
        mouth_img_filenames = None
        exp_filenames = None
        global_bunch = None
        upper_bunch = None
        middle_bunch = None
        bottom_bunch = None

        return global_accuracy, conf_mat, avg_acc, precision, recall, fscore, support