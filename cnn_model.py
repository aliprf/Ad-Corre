from config import DatasetName, AffectnetConf, InputDataSize, LearningConfig
# from hg_Class import HourglassNet

import tensorflow as tf
# from tensorflow import keras
# from skimage.transform import resize
from keras.models import Model

from keras.applications import mobilenet_v2, mobilenet, resnet50, densenet, resnet
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, \
    Dropout, ReLU, Concatenate, Input, GlobalMaxPool2D, LeakyReLU, Softmax, ELU

class CNNModel:
    def get_model(self, arch, num_of_classes, weights):
        if arch == 'resnet':
            model = self._create_resnetemb(num_of_classes,
                                           num_of_embeddings=LearningConfig.num_embeddings,
                                           input_shape=(InputDataSize.image_input_size,
                                                        InputDataSize.image_input_size, 3),
                                           weights=weights
                                           )
        if arch == 'xcp':
            model = self._create_Xception_l2(num_of_classes,
                                             num_of_embeddings=LearningConfig.num_embeddings,
                                             input_shape=(InputDataSize.image_input_size,
                                                          InputDataSize.image_input_size, 3),
                                             weights=weights
                                             )

        return model

    def _create_resnetemb(self, num_of_classes, input_shape, weights, num_of_embeddings):
        resnet_model = resnet.ResNet50(
            input_shape=input_shape,
            include_top=True,
            weights='imagenet',
            # weights=None,
            input_tensor=None,
            pooling=None)
        resnet_model.layers.pop()

        avg_pool = resnet_model.get_layer('avg_pool').output  # 2048
        ''''''
        embeddings = []
        for i in range(num_of_embeddings):
            emb = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(avg_pool)
            emb_l2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(emb)
            embeddings.append(emb_l2)

        if num_of_embeddings > 1:
            fused = tf.keras.layers.Concatenate(axis=1)([embeddings[i] for i in range(num_of_embeddings)])
        else:
            fused = embeddings[0]

        fused = Dropout(rate=0.5)(fused)

        '''out'''
        out_categorical = Dense(num_of_classes,
                                activation='softmax',
                                name='out')(fused)

        inp = [resnet_model.input]

        revised_model = Model(inp, [out_categorical] + [embeddings[i] for i in range(num_of_embeddings)])
        revised_model.summary()
        '''save json'''
        model_json = revised_model.to_json()

        with open("./model_archs/resnetemb.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def _create_Xception_l2(self, num_of_classes, num_of_embeddings, input_shape, weights):
        xception_model = tf.keras.applications.Xception(
            include_top=False,
            # weights=None,
            input_tensor=None,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None,
            classes=num_of_classes
        )

        act_14 = xception_model.get_layer('block14_sepconv2_act').output
        avg_pool = GlobalAveragePooling2D()(act_14)

        embeddings = []
        for i in range(num_of_embeddings):
            emb = tf.keras.layers.Dense(LearningConfig.embedding_size, activation=None)(avg_pool)
            emb_l2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(emb)

            embeddings.append(emb_l2)
        if num_of_embeddings > 1:
            fused = tf.keras.layers.Concatenate(axis=1)([embeddings[i] for i in range(num_of_embeddings)])
        else:
            fused = embeddings[0]
        fused = Dropout(rate=0.5)(fused)

        '''out'''
        out_categorical = Dense(num_of_classes,
                                activation='softmax',
                                name='out')(fused)

        inp = [xception_model.input]

        revised_model = Model(inp, [out_categorical] + [embeddings[i] for i in range(num_of_embeddings)])
        revised_model.summary()
        '''save json'''
        model_json = revised_model.to_json()

        with open("./model_archs/xcp_embedding.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model