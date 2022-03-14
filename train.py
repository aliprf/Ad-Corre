
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import csv
from skimage.io import imread
import pickle
from sklearn.metrics import accuracy_score
import os
import time

from AffectNetClass import AffectNet
from RafdbClass import RafDB
from FerPlusClass import FerPlus

from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig, DatasetType, RafDBConf, FerPlusConf
from cnn_model import CNNModel
from custom_loss import CustomLosses
from data_helper import DataHelper
from dataset_class import CustomDataset


class TrainModel:
    def __init__(self, dataset_name, ds_type, weights='imagenet', lr=1e-3, aug=True):
        self.dataset_name = dataset_name
        self.ds_type = ds_type
        self.weights = weights
        self.lr = lr

        self.base_lr = 1e-5
        self.max_lr = 5e-4
        if dataset_name == DatasetName.fer2013:
            self.drop = 0.1
            self.epochs_drop = 5
            if aug:
                self.img_path = FerPlusConf.aug_train_img_path
                self.annotation_path = FerPlusConf.aug_train_annotation_path
                self.masked_img_path = FerPlusConf.aug_train_masked_img_path
            else:
                self.img_path = FerPlusConf.no_aug_train_img_path
                self.annotation_path = FerPlusConf.no_aug_train_annotation_path

            self.val_img_path = FerPlusConf.test_img_path
            self.val_annotation_path = FerPlusConf.test_annotation_path
            self.eval_masked_img_path = FerPlusConf.test_masked_img_path
            self.num_of_classes = 7
            self.num_of_samples = None

        elif dataset_name == DatasetName.rafdb:
            self.drop = 0.1
            self.epochs_drop = 5

            if aug:
                self.img_path = RafDBConf.aug_train_img_path
                self.annotation_path = RafDBConf.aug_train_annotation_path
                self.masked_img_path = RafDBConf.aug_train_masked_img_path
            else:
                self.img_path = RafDBConf.no_aug_train_img_path
                self.annotation_path = RafDBConf.no_aug_train_annotation_path

            self.val_img_path = RafDBConf.test_img_path
            self.val_annotation_path = RafDBConf.test_annotation_path
            self.eval_masked_img_path = RafDBConf.test_masked_img_path
            self.num_of_classes = 7
            self.num_of_samples = None

        elif dataset_name == DatasetName.affectnet:
            self.drop = 0.1
            self.epochs_drop = 5

            if ds_type == DatasetType.train:
                self.img_path = AffectnetConf.aug_train_img_path
                self.annotation_path = AffectnetConf.aug_train_annotation_path
                self.masked_img_path = AffectnetConf.aug_train_masked_img_path
                self.val_img_path = AffectnetConf.eval_img_path
                self.val_annotation_path = AffectnetConf.eval_annotation_path
                self.eval_masked_img_path = AffectnetConf.eval_masked_img_path
                self.num_of_classes = 8
                self.num_of_samples = AffectnetConf.num_of_samples_train
            elif ds_type == DatasetType.train_7:
                if aug:
                    self.img_path = AffectnetConf.aug_train_img_path_7
                    self.annotation_path = AffectnetConf.aug_train_annotation_path_7
                    self.masked_img_path = AffectnetConf.aug_train_masked_img_path_7
                else:
                    self.img_path = AffectnetConf.no_aug_train_img_path_7
                    self.annotation_path = AffectnetConf.no_aug_train_annotation_path_7

                self.val_img_path = AffectnetConf.eval_img_path_7
                self.val_annotation_path = AffectnetConf.eval_annotation_path_7
                self.eval_masked_img_path = AffectnetConf.eval_masked_img_path_7
                self.num_of_classes = 7
                self.num_of_samples = AffectnetConf.num_of_samples_train_7

    def train(self, arch, weight_path):
        """"""

        '''create loss'''
        c_loss = CustomLosses()

        '''create summary writer'''
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        start_train_date = datetime.now().strftime("%Y%m%d-%H%M%S")

        '''making models'''
        model = self.make_model(arch=arch, w_path=weight_path)
        '''create save path'''
        if self.dataset_name == DatasetName.affectnet:
            save_path = AffectnetConf.weight_save_path + start_train_date + '/'
        elif self.dataset_name == DatasetName.rafdb:
            save_path = RafDBConf.weight_save_path + start_train_date + '/'
        elif self.dataset_name == DatasetName.fer2013:
            save_path = FerPlusConf.weight_save_path + start_train_date + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        '''create sample generator'''
        dhp = DataHelper()

        '''     Train   Generator'''
        img_filenames, exp_filenames = dhp.create_generator_full_path(img_path=self.img_path,
                                                                      annotation_path=self.annotation_path)
        '''create dataset'''
        cds = CustomDataset()
        ds = cds.create_dataset(img_filenames=img_filenames,
                                anno_names=exp_filenames,
                                is_validation=False)

        '''create train configuration'''
        step_per_epoch = len(img_filenames) // LearningConfig.batch_size
        gradients = None
        virtual_step_per_epoch = LearningConfig.virtual_batch_size // LearningConfig.batch_size

        '''create optimizer'''
        optimizer = tf.keras.optimizers.Adam(self.lr, decay=1e-5)

        '''start train:'''
        all_gt_exp = []
        all_pr_exp = []

        for epoch in range(LearningConfig.epochs):
            ce_weight = 2
            batch_index = 0

            for img_batch, exp_batch in ds:
                '''since the calculation of the confusion matrix will be time-consuming,
                    we only save 1000 labels each time. Moreover, this help us to be more qiuck on updates
                    '''
                all_gt_exp, all_pr_exp = self._update_all_labels_arrays(all_gt_exp, all_pr_exp)
                '''load annotation and images'''
                '''squeeze'''
                exp_batch = exp_batch[:, -1]
                img_batch = img_batch[:, -1, :, :]

                '''train step'''
                step_gradients, all_gt_exp, all_pr_exp = self.train_step(epoch=epoch, step=batch_index,
                                                                         total_steps=step_per_epoch,
                                                                         img_batch=img_batch,
                                                                         anno_exp=exp_batch,
                                                                         model=model, optimizer=optimizer,
                                                                         c_loss=c_loss,
                                                                         ce_weight=ce_weight,
                                                                         summary_writer=summary_writer,
                                                                         all_gt_exp=all_gt_exp,
                                                                         all_pr_exp=all_pr_exp)
                batch_index += 1

            '''evaluating part'''
            global_accuracy, conf_mat, avg_acc = self._eval_model(model=model)
            '''save weights'''
            save_name = save_path + '_' + str(epoch) + '_' + self.dataset_name + '_AC_' + str(global_accuracy)
            model.save(save_name + '.h5')
            self._save_confusion_matrix(conf_mat, save_name + '.txt')

    def train_step(self, epoch, step, total_steps, model, ce_weight,
                   img_batch, anno_exp, optimizer, summary_writer, c_loss, all_gt_exp, all_pr_exp):
        with tf.GradientTape() as tape:
            pr_data = model([img_batch], training=True)
            exp_pr_vec = pr_data[0]
            embeddings = pr_data[1:]

            bs_size = tf.shape(exp_pr_vec, out_type=tf.dtypes.int64)[0]

            loss_exp, accuracy = c_loss.cross_entropy_loss(y_pr=exp_pr_vec, y_gt=anno_exp,
                                                           num_classes=self.num_of_classes,
                                                           ds_name=self.dataset_name)

            '''Feature difference loss'''
            # embedding_similarity_loss = 0
            embedding_similarity_loss = c_loss.embedding_loss_distance(embeddings=embeddings)

            '''update confusion matrix'''
            exp_pr = tf.constant([np.argmax(exp_pr_vec[i]) for i in range(bs_size)], dtype=tf.dtypes.int64)
            tr_conf_matrix, all_gt_exp, all_pr_exp = c_loss.update_confusion_matrix(anno_exp,  # real labels
                                                                                    exp_pr,  # real labels
                                                                                    all_gt_exp,
                                                                                    all_pr_exp)
            ''' correlation between the embeddings'''
            correlation_loss = c_loss.correlation_loss_multi(embeddings=embeddings,
                                                             exp_gt_vec=anno_exp,
                                                             exp_pr_vec=exp_pr_vec,
                                                             tr_conf_matrix=tr_conf_matrix)
            '''mean loss'''
            mean_correlation_loss = c_loss.mean_embedding_loss_distance(embeddings=embeddings,
                                                                        exp_gt_vec=anno_exp,
                                                                        exp_pr_vec=exp_pr_vec,
                                                                        num_of_classes=self.num_of_classes)

            lamda_param = 50
            loss_total = lamda_param * loss_exp + \
                         embedding_similarity_loss + \
                         correlation_loss + \
                         mean_correlation_loss

            # '''calculate gradient'''
        gradients_of_model = tape.gradient(loss_total, model.trainable_variables)
        # '''apply Gradients:'''
        optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
        # '''printing loss Values: '''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step) + '/' + str(total_steps),
                 ' -> : accuracy: ', accuracy,
                 ' -> : loss_total: ', loss_total,
                 ' -> : loss_exp: ', loss_exp,
                 ' -> : embedding_similarity_loss: ', embedding_similarity_loss,
                 ' -> : correlation_loss: ', correlation_loss,
                 ' -> : mean_correlation_loss: ', mean_correlation_loss)
        with summary_writer.as_default():
            tf.summary.scalar('loss_total', loss_total, step=epoch)
            tf.summary.scalar('loss_exp', loss_exp, step=epoch)
            tf.summary.scalar('correlation_loss', correlation_loss, step=epoch)
            tf.summary.scalar('mean_correlation_loss', mean_correlation_loss, step=epoch)
            tf.summary.scalar('embedding_similarity_loss', embedding_similarity_loss, step=epoch)
        return gradients_of_model, all_gt_exp, all_pr_exp

    def train_step_old(self, epoch, step, total_steps, model, ce_weight,
                       img_batch, anno_exp, optimizer, summary_writer, c_loss, all_gt_exp, all_pr_exp):
        with tf.GradientTape() as tape:
            # '''create annotation_predicted'''
            # exp_pr, embedding = model([img_batch], training=True)
            exp_pr_vec, embedding_class, embedding_mean, embedding_var = model([img_batch], training=True)

            bs_size = tf.shape(exp_pr_vec, out_type=tf.dtypes.int64)[0]
            # # '''CE loss'''
            loss_exp, accuracy = c_loss.cross_entropy_loss(y_pr=exp_pr_vec, y_gt=anno_exp,
                                                           num_classes=self.num_of_classes,
                                                           ds_name=self.dataset_name)
            #
            loss_cls_mean, loss_cls_var, loss_mean_var = c_loss.embedding_loss_distance(
                embedding_class=embedding_class,
                embedding_mean=embedding_mean,
                embedding_var=embedding_var,
                bs_size=bs_size)
            feature_diff_loss = loss_cls_mean + loss_cls_var + loss_mean_var

            # correlation between the class_embeddings
            cor_loss, all_gt_exp, all_pr_exp = c_loss.correlation_loss(embedding=embedding_class,  # distribution
                                                                       exp_gt_vec=anno_exp,
                                                                       exp_pr_vec=exp_pr_vec,
                                                                       num_of_classes=self.num_of_classes,
                                                                       all_gt_exp=all_gt_exp,
                                                                       all_pr_exp=all_pr_exp)
            # correlation between the mean_emb_cor_loss
            mean_emb_cor_loss, mean_emb_kl_loss = c_loss.mean_embedding_loss(embedding=embedding_mean,
                                                                             exp_gt_vec=anno_exp,
                                                                             exp_pr_vec=exp_pr_vec,
                                                                             num_of_classes=self.num_of_classes)
            mean_loss = mean_emb_cor_loss + 10 * mean_emb_kl_loss

            var_emb_cor_loss, var_emb_kl_loss = c_loss.variance_embedding_loss(embedding=embedding_var,
                                                                               exp_gt_vec=anno_exp,
                                                                               exp_pr_vec=exp_pr_vec,
                                                                               num_of_classes=self.num_of_classes)
            var_loss = var_emb_cor_loss + 10 * var_emb_kl_loss
            # '''total:'''
            loss_total = 100 * loss_exp + cor_loss + 10 * feature_diff_loss + mean_loss + var_loss

        # '''calculate gradient'''
        gradients_of_model = tape.gradient(loss_total, model.trainable_variables)
        # '''apply Gradients:'''
        optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
        # '''printing loss Values: '''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step) + '/' + str(total_steps),
                 ' -> : accuracy: ', accuracy,
                 ' -> : loss_total: ', loss_total,
                 ' -> : loss_exp: ', loss_exp,
                 ' -> : cor_loss: ', cor_loss,
                 ' -> : feature_loss: ', feature_diff_loss,
                 ' -> : mean_loss: ', mean_loss,
                 ' -> : var_loss: ', var_loss)

        with summary_writer.as_default():
            tf.summary.scalar('loss_total', loss_total, step=epoch)
            tf.summary.scalar('loss_exp', loss_exp, step=epoch)
            tf.summary.scalar('loss_correlation', cor_loss, step=epoch)
        return gradients_of_model, all_gt_exp, all_pr_exp

    def _eval_model(self, model):
        """"""
        '''first we need to create the 4 bunch here: '''

        '''for Affectnet, we need to calculate accuracy of each label and then total avg accuracy:'''
        global_accuracy = 0
        avg_acc = 0
        conf_mat = []
        if self.dataset_name == DatasetName.affectnet:
            if self.ds_type == DatasetType.train:
                affn = AffectNet(ds_type=DatasetType.eval)
            else:
                affn = AffectNet(ds_type=DatasetType.eval_7)
            global_accuracy, conf_mat, avg_acc, precision, recall, fscore, support = \
                affn.test_accuracy(model=model)
        elif self.dataset_name == DatasetName.rafdb:
            rafdb = RafDB(ds_type=DatasetType.test)
            global_accuracy, conf_mat, avg_acc, precision, recall, fscore, support = rafdb.test_accuracy(model=model)
        elif self.dataset_name == DatasetName.fer2013:
            ferplus = FerPlus(ds_type=DatasetType.test)
            global_accuracy, conf_mat, avg_acc, precision, recall, fscore, support = ferplus.test_accuracy(model=model)
        print("================== global_accuracy =====================")
        print(global_accuracy)
        print("================== Average Accuracy =====================")
        print(avg_acc)
        print("================== Confusion Matrix =====================")
        print(conf_mat)
        return global_accuracy, conf_mat, avg_acc

    def make_model(self, arch, w_path):
        cnn = CNNModel()
        model = cnn.get_model(arch=arch, num_of_classes=LearningConfig.num_classes, weights=self.weights)
        if w_path is not None:
            model.load_weights(w_path)
        return model

    def _save_confusion_matrix(self, conf_mat, save_name):
        f = open(save_name, "a")
        print(save_name)
        f.write(np.array_str(conf_mat))
        f.close()

    def _update_all_labels_arrays(self, all_gt_exp, all_pr_exp):
        if len(all_gt_exp) < LearningConfig.labels_history_frame:
            return all_gt_exp, all_pr_exp
        else:  # remove the first batch:
            return all_gt_exp[LearningConfig.batch_size:], all_pr_exp[LearningConfig.batch_size:]