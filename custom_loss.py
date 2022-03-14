import tensorflow as tf
import numpy as np
from config import DatasetName, ExpressionCodesRafdb, ExpressionCodesAffectnet
from keras import backend as K
from sklearn.metrics import confusion_matrix
import time
from config import LearningConfig


class CustomLosses:

    def embedding_loss_distance(self, embeddings):

        """
        for each item in batch: calculate the correlation between all the embeddings
        :param embeddings:
        :return:
        """
        '''correlation'''
        emb_len = len(embeddings)
        ''' emb_num, bs, emb_size: '''
        embeddings = tf.cast(embeddings, dtype=tf.dtypes.float32)
        loss = tf.cast([np.corrcoef(embeddings[:, i, :]) for i in range(LearningConfig.batch_size)],
                       dtype=tf.dtypes.float32)
        embedding_similarity_loss = tf.reduce_mean((1 - tf.eye(emb_len)) *  # ignore the affect of the diagonal
                                                   (1 + np.array(loss)))  # the loss -> more correlation, the better
        return embedding_similarity_loss

    def mean_embedding_loss_distance(self, embeddings, exp_gt_vec, exp_pr_vec, num_of_classes):
        """
        calculate the mean distribution for each class, and force the mean_embedding to be the same
        :param embedding: bs * embedding_size
        :param exp_gt_vec: bs
        :param exp_pr_vec: bs * num_of_classes
        :param num_of_classes:
        :return:
        """
        # bs_size = tf.shape(exp_pr_vec, out_type=tf.dtypes.int64)[0]
        #  7 * bs: for each class - which we have 7 class, not the embeddings -, is the sample belongs to it we put 1
        # else we put 0 .  THE SAME FOR ALL embeddings
        c_map = np.array([tf.cast(tf.where(exp_gt_vec == i, 1.0, K.epsilon()), dtype=tf.dtypes.float32)
                          for i in range(num_of_classes)])  # 7 * bs

        # calculate class-related mean embedding: num_of_classes * embedding_size
        # 7 embedding, and 7 classes and each class has an embedding mean of 256 -> 7,7,256
        mean_embeddings = np.array([[np.average(embeddings[k], axis=0, weights=c_map[i, :])
                                     for i in range(num_of_classes)]
                                    for k in range(len(embeddings))])  # 7:embedding,7:class, 256:size

        #  the correlation between each mean_embedding should be low:
        mean_emb_correlation_loss = tf.reduce_mean([(1 - tf.eye(num_of_classes)) *  # zero the diagonal
                                                    (1 + tf.cast(np.corrcoef(mean_embeddings[k, :, :]),
                                                                 dtype=tf.dtypes.float32))
                                                    for k in range(len(embeddings))])

        # the correlation between each mean_embedding should be low:
        return mean_emb_correlation_loss

    def mean_embedding_loss(self, embedding, exp_gt_vec, exp_pr_vec, num_of_classes):
        """
        calculate the mean distribution for each class, and force the mean_embedding to be the same
        :param embedding: bs * embedding_size
        :param exp_gt_vec: bs
        :param exp_pr_vec: bs * num_of_classes
        :param num_of_classes:
        :return:
        """
        kl = tf.keras.losses.KLDivergence()
        bs_size = tf.shape(exp_pr_vec, out_type=tf.dtypes.int64)[0]
        # calculate class maps: num_of_classes * bs
        c_map = np.array([tf.cast(tf.where(exp_gt_vec == i, 1, 0), dtype=tf.dtypes.int8)
                          for i in range(num_of_classes)])  # 7 * bs
        # calculate class-related mean embedding: num_of_classes * embedding_size
        mean_embeddings = np.array([np.average(embedding, axis=0, weights=c_map[i, :])
                                    if np.sum(c_map[i, :]) > 0 else np.zeros(LearningConfig.embedding_size)
                                    for i in range(num_of_classes)]) + \
                          K.epsilon()  # added as a bias to get ride of zeros

        # calculate loss:
        #   1 -> the correlation between each mean_embedding should be low:
        mean_emb_correlation_loss = tf.reduce_mean((1 - tf.eye(num_of_classes)) *
                                                   (1 + tf.cast(np.corrcoef(mean_embeddings), dtype=tf.dtypes.float32)))
        #   2 -> the KL-divergence between the mean distribution of each class and the related
        #   embeddings should be low. Accordingly, we lead the network towards learning the mean distribution
        mean_emb_batch = tf.cast([np.array(mean_embeddings)[i] for i in np.argmax(c_map.T, axis=1)],
                                 dtype=tf.dtypes.float32)
        emb_kl_loss = kl(y_true=mean_emb_batch, y_pred=embedding)

        return mean_emb_correlation_loss, emb_kl_loss

    def variance_embedding_loss(self, embedding, exp_gt_vec, exp_pr_vec, num_of_classes):
        """
        calculate the variance of the distribution for each class, and force the mean_embedding to be the same
        :param embedding:
        :param exp_gt_vec:
        :param exp_pr_vec:
        :param num_of_classes:
        :return:
        """
        kl = tf.keras.losses.KLDivergence()
        bs_size = tf.shape(exp_pr_vec, out_type=tf.dtypes.int64)[0]
        #
        c_map = np.array([tf.cast(tf.where(exp_gt_vec == i, 1, 0), dtype=tf.dtypes.int8)
                          for i in range(num_of_classes)])  # 7 * bs
        # calculate class-related var embedding: num_of_classes * embedding_size
        var_embeddings = np.array([tf.math.reduce_std(tf.math.multiply(embedding,
                                                                       tf.repeat(tf.expand_dims(
                                                                           tf.cast(c_map[i, :],
                                                                                   dtype=tf.dtypes.float32), -1),
                                                                           LearningConfig.embedding_size, axis=-1, )),
                                                      axis=0)
                                   for i in range(num_of_classes)]) \
                         + K.epsilon()  # added as a bias to get ride of zeros

        # calculate loss:
        #   1 -> the correlation between each mean_embedding should be low:
        var_emb_correlation_loss = tf.reduce_mean((1.0 - tf.eye(num_of_classes)) *
                                                  (1.0 + tf.cast(np.cov(var_embeddings), dtype=tf.dtypes.float32)))
        #   embeddings should be low. Accordingly, we lead the network towards learning the mean distribution
        var_emb_batch = tf.cast([np.array(var_embeddings)[i] for i in np.argmax(c_map.T, axis=1)],
                                dtype=tf.dtypes.float32)
        emb_kl_loss = abs(kl(y_true=var_emb_batch, y_pred=embedding))
        return var_emb_correlation_loss, emb_kl_loss

    def correlation_loss(self, embedding, exp_gt_vec, exp_pr_vec, tr_conf_matrix):
        bs_size = tf.shape(exp_pr_vec, out_type=tf.dtypes.int64)[0]
        # convert from sigmoid to labels to real classes:
        exp_pr = tf.constant([np.argmax(exp_pr_vec[i]) for i in range(bs_size)], dtype=tf.dtypes.int64)
        # Cov matrix
        phi_correlation_matrix = tf.cast(np.corrcoef(embedding), dtype=tf.dtypes.float32)  # bs * bs

        elems_col = tf.repeat(tf.expand_dims(exp_gt_vec, 0), repeats=[bs_size], axis=0)
        elems_row = tf.repeat(tf.expand_dims(exp_gt_vec, -1), repeats=[bs_size], axis=-1)
        delta = elems_row - elems_col
        omega_matrix = tf.cast(tf.where(delta == 0, 1, -1), dtype=tf.dtypes.float32)
        # creating the adaptive weights
        adaptive_weight = self._create_adaptive_correlation_weights(bs_size=bs_size,
                                                                    exp_gt_vec=exp_gt_vec,  # real labels
                                                                    exp_pr=exp_pr,  # real labels
                                                                    conf_mat=tr_conf_matrix)
        # calculate correlation loss
        cor_loss = tf.reduce_mean(adaptive_weight * tf.abs(omega_matrix - phi_correlation_matrix))
        return cor_loss

    def correlation_loss_multi(self, embeddings, exp_gt_vec, exp_pr_vec, tr_conf_matrix):
        """
        here, we consider only one embedding and so want to make the embeddings of the same classes be similar
        while the ones from different classes are different.
        :param embeddings:
        :param exp_gt_vec:
        :param exp_pr_vec:
        :param tr_conf_matrix:
        :return:
        """
        bs_size = tf.shape(exp_pr_vec, out_type=tf.dtypes.int64)[0]
        exp_pr = tf.constant([np.argmax(exp_pr_vec[i]) for i in range(bs_size)], dtype=tf.dtypes.int64)

        phi_correlation_matrices = [tf.cast(np.corrcoef(embeddings[i]), dtype=tf.dtypes.float32)
                                    for i in range(len(embeddings))]  # cls * bs * bs
        #
        elems_col = tf.repeat(tf.expand_dims(exp_gt_vec, 0), repeats=[bs_size], axis=0)
        elems_row = tf.repeat(tf.expand_dims(exp_gt_vec, -1), repeats=[bs_size], axis=-1)
        delta = elems_row - elems_col
        omega_matrix = tf.repeat(tf.expand_dims(tf.cast(tf.where(delta == 0, 1, -1),
                                                        dtype=tf.dtypes.float32), axis=0),
                                 repeats=len(embeddings), axis=0)
        cor_loss = tf.reduce_mean(tf.abs(omega_matrix - phi_correlation_matrices))

        return cor_loss

    def _create_adaptive_correlation_weights(self, bs_size, exp_gt_vec, exp_pr, conf_mat):
        """
        creating the weights
        :param exp_gt_vec: real int labels
        :param exp_pr_vec: one_hot labels
        :param conf_mat: confusion matrix which is normalized over the rows(
                        ground-truths with respect to the number of corresponding classes)
        :return: a bath_size * bath_size matrix containing weights. The diameter of the matrix is zero
        """
        '''new'''
        tf_identity = tf.eye(bs_size)
        # weight based on the correct section of the conf_matrix
        '''
        1 : - conf_mat[exp_gt_vec[i], exp_gt_vec[i]] : sum of all the missed values=> the better the performance of
        the model on a label, the smaller the weight  
        '''
        correct_row_base_weight = tf.repeat(tf.expand_dims(
            tf.map_fn(fn=lambda i: 1 - conf_mat[i, i], elems=exp_gt_vec)  # map
            , 0),  # expand_dims
            repeats=[bs_size], axis=0)  # repeat

        correct_col_base_weight = tf.einsum('ab->ba', correct_row_base_weight)
        correct_weight = correct_row_base_weight + correct_col_base_weight
        adaptive_weight = tf.cast((correct_weight), dtype=tf.dtypes.float32)
        adaptive_weight = 1 + adaptive_weight  # we don't want the weights to be zero (correct prediction)
        adaptive_weight = (1 - tf_identity) * adaptive_weight  # remove the main diagon
        return adaptive_weight

    def update_confusion_matrix(self, exp_gt_vec, exp_pr,
                                all_gt_exp, all_pr_exp):
        # adding to the previous predicted items:
        all_pr_exp += np.array(exp_pr).tolist()
        all_gt_exp += np.array(exp_gt_vec).tolist()
        # calculate confusion matrix:
        conf_mat = confusion_matrix(y_true=all_gt_exp, y_pred=all_pr_exp, normalize='true',
                                    labels=[0, 1, 2, 3, 4, 5, 6])
        return conf_mat, all_gt_exp, all_pr_exp

    def cross_entropy_loss(self, y_gt, y_pr, num_classes, ds_name):
        y_gt_oh = tf.one_hot(y_gt, depth=num_classes)
        ''' manual weighted CE'''
        y_pred = y_pr
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        loss = -tf.reduce_mean(y_gt_oh * tf.math.log(y_pred))
        '''accuracy'''
        accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pr, y_gt_oh))
        return loss, accuracy