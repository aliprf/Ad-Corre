from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, ExpressionCodesAffectnet
from config import LearningConfig, InputDataSize, DatasetName, FerPlusConf, DatasetType

import numpy as np
import os
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray, savez_compressed
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
from shutil import copyfile
from dataset_class import CustomDataset
from sklearn.metrics import precision_recall_fscore_support as score


class FerPlus:

    def __init__(self, ds_type):
        """we set the parameters needed during the whole class:
        """
        self.ds_type = ds_type
        if ds_type == DatasetType.train:
            self.img_path = FerPlusConf.no_aug_train_img_path
            self.anno_path = FerPlusConf.no_aug_train_annotation_path
            self.img_path_aug = FerPlusConf.aug_train_img_path
            self.anno_path_aug = FerPlusConf.aug_train_annotation_path
            self.masked_img_path = FerPlusConf.aug_train_masked_img_path
            self.orig_image_path = FerPlusConf.orig_image_path_train

        elif ds_type == DatasetType.test:
            self.img_path = FerPlusConf.test_img_path
            self.anno_path = FerPlusConf.test_annotation_path
            self.img_path_aug = FerPlusConf.test_img_path
            self.anno_path_aug = FerPlusConf.test_annotation_path
            self.masked_img_path = FerPlusConf.test_masked_img_path
            self.orig_image_path = FerPlusConf.orig_image_path_train

    def create_from_orig(self):
        print('create_from_orig & relabel to affectNetLike--->')
        """
        labels are from 1-7, but we save them from 0 to 6
        :param ds_type:
        :return:
        """

        '''read the text file, and save exp, and image'''
        dhl = DataHelper()

        exp_affectnet_like_lbls = [6, 5, 4, 1, 0, 2, 3]
        lbl_affectnet_like_lbls = ['angry/', 'disgust/', 'fear/', 'happy/', 'neutral/', 'sad/', 'surprise/']

        for exp_index in range(len(lbl_affectnet_like_lbls)):
            exp_prefix = lbl_affectnet_like_lbls[exp_index]
            for i, file in tqdm(enumerate(os.listdir(self.orig_image_path + exp_prefix))):
                if file.endswith(".jpg") or file.endswith(".png"):
                    img_source_address = self.orig_image_path + exp_prefix + file
                    img_dest_address = self.img_path + file
                    exp_dest_address = self.anno_path + file[:-4]
                    exp = exp_affectnet_like_lbls[exp_index]

                    img = np.array(Image.open(img_source_address))
                    res_img = resize(img, (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                                     anti_aliasing=True)

                    im = Image.fromarray(np.round(res_img * 255.0).astype(np.uint8))
                    '''save image'''
                    im.save(img_dest_address)
                    '''save annotation'''
                    np.save(exp_dest_address + '_exp', exp)

    def test_accuracy(self, model, print_samples=False):
        print('FER: test_accuracy')
        dhp = DataHelper()
        '''create batches'''
        img_filenames, exp_filenames = dhp.create_generator_full_path(
            img_path=self.img_path,
            annotation_path=self.anno_path, label=None)
        print(len(img_filenames))
        exp_pr_lbl = []
        exp_gt_lbl = []

        cds = CustomDataset()
        ds = cds.create_dataset(img_filenames=img_filenames,
                                anno_names=exp_filenames,
                                is_validation=True,
                                ds=DatasetName.fer2013)

        batch_index = 0
        print('FER: loading test ds')
        for img_batch, exp_gt_b in tqdm(ds):
            '''predict on batch'''
            exp_gt_b = exp_gt_b[:, -1]
            img_batch = img_batch[:, -1, :, :]

            pr_data = model.predict_on_batch([img_batch])
            probab_exp_pr_b = pr_data[0]

            scores_b = np.array([tf.nn.softmax(probab_exp_pr_b[i]) for i in range(len(probab_exp_pr_b))])
            exp_pr_b = np.array([np.argmax(scores_b[i]) for i in range(len(probab_exp_pr_b))])

            if print_samples:
                for i in range(len(exp_pr_b)):
                    dhp.test_image_print_exp(str(i)+str(batch_index+1), np.array(img_batch[i]),
                                             np.int8(exp_gt_b[i]), np.int8(exp_pr_b[i]))

            exp_pr_lbl += np.array(exp_pr_b).tolist()
            exp_gt_lbl += np.array(exp_gt_b).tolist()
            batch_index += 1
        exp_pr_lbl = np.float64(np.array(exp_pr_lbl))
        exp_gt_lbl = np.float64(np.array(exp_gt_lbl))

        global_accuracy = accuracy_score(exp_gt_lbl, exp_pr_lbl)
        precision, recall, fscore, support = score(exp_gt_lbl, exp_pr_lbl)
        conf_mat = confusion_matrix(exp_gt_lbl, exp_pr_lbl, normalize='true')
        # conf_mat = tf.math.confusion_matrix(exp_gt_lbl, exp_pr_lbl, num_classes=7)
        avg_acc = np.mean([conf_mat[i,i] for i in range(7)])

        ds = None
        face_img_filenames = None
        eyes_img_filenames = None
        nose_img_filenames = None
        mouth_img_filenames = None
        exp_filenames = None

        return global_accuracy, conf_mat, avg_acc, precision, recall, fscore, support
