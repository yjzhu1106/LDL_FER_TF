import tensorflow as tf
import os

from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
'''
重写正常模型,方便去input和output

'''


from model.resnet50 import ResNet50


class AdaptiveSimNet(tf.keras.Model):
    def __init__(self, num_neighbors=4, feature_dim=512):
        super(AdaptiveSimNet, self).__init__()
        self.num_neighbors = num_neighbors

        # self.mha = MultiHeadAttention(512, 1)

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(1, activation=None),
            tf.keras.layers.Lambda(lambda x: tf.maximum(x, -1.3)),
            tf.keras.layers.Activation('sigmoid')
        ])

    def call(self, feat, feat_aux, training=False):
        '''
        feat: shape (B,D) - D is the hidden dimension
        feat_aux: shape (B,K,D) - K number of neighbors
        return weighting score for neighbors: shape (B,K)
        '''
        # feat_aux = self.mha(feat_aux, training=training)

        B, K, D = feat_aux.shape


        feat_repeat = tf.keras.layers.RepeatVector(K)(feat)  # shape (B,K, D)
        x = tf.keras.layers.Concatenate(axis=-1)([feat_repeat, feat_aux])  # shape (B,K,2*D)

        scores = tf.squeeze(self.encoder(x, training=training), axis=-1)
        # scores = tf.sigmoid(tf.reduce_sum(tf.multiply(feat_repeat,feat_aux),axis=-1)/tf.sqrt(512.0))

        return scores
        # return tf.maximum(scores, 0.2)

def adaptiveSimNet(input_shape=(112, 112, 3), num_neighbors=4, feature_dim=512):


    img_input = layers.Input(shape=input_shape, name='adaptive_input_layer')
    feat = img_input[0]
    feat_aux = img_input[1:]

    B, K, D = feat_aux.shape

    feat_repeat = tf.keras.layers.RepeatVector(K)(feat)  # shape (B,K, D)
    x = tf.keras.layers.Concatenate(axis=-1)([feat_repeat, feat_aux])  # shape (B,K,2*D)



    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.HeNormal(),name="encoder1"),
        tf.keras.layers.BatchNormalization(name="encoder2"),
        tf.keras.layers.ReLU(name="encoder3"),
        tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.HeNormal(), name="encoder4"),
        tf.keras.layers.BatchNormalization(name="encoder5"),
        tf.keras.layers.Activation('relu', name="encoder6"),
        tf.keras.layers.Dense(1, activation=None, name="encoder7"),
        tf.keras.layers.Lambda(lambda x: tf.maximum(x, -1.3), name="encoder8"),
        tf.keras.layers.Activation('sigmoid', name="encoder9")
    ])(img_input)

    scores = tf.squeeze(encoder, axis=-1)
    model = training.Model(img_input, scores, name="adaptiveSimNet")
    return model






def multi_channel(input_shape=(4, 4, 2048), filters=2048):
    img_input = layers.Input(shape=input_shape, name='multi_channel_input_layer')

    f11 = tf.keras.layers.Conv2D(filters, 1, dilation_rate=1, padding='same', use_bias=False, name='dilated_1_conv')(img_input)
    f12 = tf.keras.layers.BatchNormalization(name='normal_1_layer')(f11)
    f1 = tf.keras.layers.Activation('relu', name='activation_1_relu')(f12)

    f21 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=3, padding='same', use_bias=False, name='dilated_2_conv')(img_input)
    f22 = tf.keras.layers.BatchNormalization(name='normal_2_layer')(f21)
    f2 = tf.keras.layers.Activation('relu', name='activation_2_relu')(f22)

    f31 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=5, padding='same', use_bias=False, name='dilated_3_conv')(img_input)
    f32 = tf.keras.layers.BatchNormalization(name='normal_3_layer')(f31)
    f3 = tf.keras.layers.Activation('relu', name='activation_3_relu')(f32)

    f41 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False, name='dilated_4_conv')(img_input)
    f42 = tf.keras.layers.BatchNormalization(name='normal_4_layer')(f41)
    f4 = tf.keras.layers.Activation('relu', name='activation_4_relu')(f42)

    # f1 = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(filters, 1, dilation_rate=1, padding='same', use_bias=False, name='dilated_1_conv'),
    #         tf.keras.layers.BatchNormalization(name='normal_1_layer'),
    #         tf.keras.layers.Activation('relu', name='activation_1_relu'),
    #     ])(img_input)
    # f2 = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(filters, 3, dilation_rate=3, padding='same', use_bias=False, name='dilated_2_conv'),
    #         tf.keras.layers.BatchNormalization(name='normal_2_layer'),
    #         tf.keras.layers.Activation('relu', name='activation_2_relu'),
    #     ])(img_input)
    # f3 = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(filters, 3, dilation_rate=5, padding='same', use_bias=False, name='dilated_3_conv'),
    #         tf.keras.layers.BatchNormalization(name='normal_3_layer'),
    #         tf.keras.layers.Activation('relu', name='activation_3_relu'),
    #     ])(img_input)
    # f4 = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False, name='dilated_4_conv'),
    #         tf.keras.layers.BatchNormalization(name='normal_4_layer'),
    #         tf.keras.layers.Activation('relu', name='activation_4_relu'),
    #     ])(img_input)

    fb1 = tf.keras.layers.Add(name='add1')([f1, f2])
    fb2 = tf.keras.layers.Add(name='add2')([f3, f4])

    wb1 = tf.keras.layers.Activation('sigmoid', name='activation_sigmoid1')(fb1)
    wb2 = tf.keras.layers.Activation('sigmoid', name='activation_sigmoid2')(fb2)

    fr1 = tf.multiply(fb1, wb1,name='m1')
    fr2 = tf.multiply(fb2, wb2,name='m2')

    yc = tf.keras.layers.AvgPool2D(pool_size=(1,1),name='avg_pool1')(fr1)
    zc = tf.keras.layers.AvgPool2D(pool_size=(1,1),name='avg_pool2')(fr2)

    yc_1 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
                           name='dilated_au_1_conv1')(yc)
    yc_2 = tf.keras.layers.BatchNormalization(name='normal_au_1_layer')(yc_1)
    yc_3 = tf.keras.layers.Activation('relu', name='activation_au_1_relu')(yc_2)
    p1 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
                           name='dilated_au_1_conv2')(yc_3)


    # p1 = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
    #                                name='dilated_au_1_conv1'),
    #         tf.keras.layers.BatchNormalization(name='normal_au_1_layer'),
    #         tf.keras.layers.Activation('relu', name='activation_au_1_relu'),
    #         tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
    #                                name='dilated_au_1_conv2'),
    #     ])(yc)

    zc_1 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
                           name='dilated_au_2_conv1')(zc)
    zc_1 = tf.keras.layers.BatchNormalization(name='normal_au_2_layer')(zc_1)
    zc_1 = tf.keras.layers.Activation('relu', name='activation_au_2_relu')(zc_1)
    p2 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
                           name='dilated_au_2_conv2')(zc_1)

    # p2 = tf.keras.Sequential([
    #         tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
    #                                name='dilated_au_2_conv1'),
    #         tf.keras.layers.BatchNormalization(name='normal_au_2_layer'),
    #         tf.keras.layers.Activation('relu', name='activation_au_2_relu'),
    #         tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
    #                                name='dilated_au_2_conv2'),
    #     ])(zc)

    p1 = tf.keras.layers.Softmax(name='softmax1')(p1)
    p2 = tf.keras.layers.Softmax(name='softmax2')(p2)

    fa1 = tf.keras.layers.Add(name='add3')([tf.multiply(f1, p1), tf.multiply(f2, p1)])
    fa2 = tf.keras.layers.Add(name='add4')([tf.multiply(f3, p2), tf.multiply(f4, p2)])

    fa = tf.keras.layers.Add(name='add5')([fa1, fa2])

    ffus = tf.keras.layers.Conv2D(filters, 1, name='last_conv')(fa)
    model = training.Model(img_input, ffus, name="multiChannel")

    return model




def expNet2(num_classes=7, pretrained='imagenet', backbone="resnet50", resnetPooling="avg",feature_dim=512,
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=7,
           classifier_activation='softmax',
           **kwargs):
    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=112,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    img_input = layers.Input(shape=input_shape, name='input_layer')
    # backbone (None, 2048)
    backbone = ResNet50(input_shape=(112,112,3), weights=pretrained, include_top=False, pooling=resnetPooling)(img_input)

    multiChannel = multi_channel(input_shape=(4, 4, 2048), filters=2048)(backbone)

    feat_map = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(multiChannel)

    x = feat_map

    out = tf.keras.layers.Dense(num_classes, activation='softmax', name='softmax',
                                        kernel_initializer=tf.keras.initializers.HeNormal(),
                                        kernel_regularizer=tf.keras.regularizers.L2(0.001)
                                        )(feat_map)
    output = [x, out, multiChannel]
    model = training.Model(img_input, output, name="LDL-MGFF")

    return model

def create_model2(config):
    model = expNet2(num_classes=config.num_classes, pretrained=config.pretrained, resnetPooling=config.resnetPooling,backbone=config.backbone)
    model(tf.ones((32, config.input_size[0], config.input_size[1], 3)))
    model_adpt = AdaptiveSimNet(config.feature_dim)

    model_adpt(tf.ones((2, config.feature_dim)), tf.ones((2, 4, config.feature_dim)))
    print("Our Model")
    print(model.summary())
    print("Our Model_adpt")
    print(model_adpt.summary())
    return model, model_adpt


if __name__=="__main__":
    # model = ExpNet(num_classes=7, pretrained="msceleb", backbone="resnet50")
    # model(tf.ones((32, 112, 112, 3)))
    # model.weighting_net(tf.ones((2, 512)), tf.ones((2, 4, 512)))
    # model.summary()

    model = expNet2(num_classes=7, pretrained=None, backbone="resnet50", resnetPooling=None)
    print(model(tf.ones((32, 112, 112, 3)))[0].shape)
    # model.weighting_net(tf.ones((2, 512)), tf.ones((2, 4, 512)))
    model.summary()







