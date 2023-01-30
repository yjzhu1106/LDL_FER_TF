import tensorflow as tf
import os

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


class MultiChannel(tf.keras.Model):
    def __init__(self, filters):
        super(MultiChannel, self).__init__()

        self.f1_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 1, dilation_rate=1, padding='same', use_bias=False, name='dilated_1_conv'),
            tf.keras.layers.BatchNormalization(name='normal_1_layer'),
            tf.keras.layers.Activation('relu', name='activation_1_relu'),
        ])
        self.f2_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, dilation_rate=3, padding='same', use_bias=False, name='dilated_2_conv'),
            tf.keras.layers.BatchNormalization(name='normal_2_layer'),
            tf.keras.layers.Activation('relu', name='activation_2_relu'),
        ])
        self.f3_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, dilation_rate=5, padding='same', use_bias=False, name='dilated_3_conv'),
            tf.keras.layers.BatchNormalization(name='normal_3_layer'),
            tf.keras.layers.Activation('relu', name='activation_3_relu'),
        ])
        self.f4_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False, name='dilated_4_conv'),
            tf.keras.layers.BatchNormalization(name='normal_4_layer'),
            tf.keras.layers.Activation('relu', name='activation_4_relu'),
        ])

        self.add_layer = tf.keras.layers.Add(name='add')
        self.multiply_layer = tf.keras.layers.Multiply(name='multiply')

        self.sigmoid_activation = tf.keras.layers.Activation('sigmoid', name='activation_sigmoid')

        # self.pool_layer = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')
        self.pool_layer_1d = tf.keras.layers.AvgPool1D(name='avg_pool')
        self.pool_layer = tf.keras.layers.AvgPool2D(pool_size=(1,1),name='avg_pool')

        self.pool_layer_3d = tf.keras.layers.AvgPool3D(name='avg_pool')


        self.attention1_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
                                   name='dilated_au_1_conv1'),
            tf.keras.layers.BatchNormalization(name='normal_au_1_layer'),
            tf.keras.layers.Activation('relu', name='activation_au_1_relu'),
            tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
                                   name='dilated_au_1_conv2'),
        ])

        self.attention1_conv_dense = tf.keras.layers.Dense(2048)
        self.attention1_conv_normal = tf.keras.layers.BatchNormalization()
        self.attention1_conv_relu = tf.keras.layers.Activation('relu'),




        self.attention2_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
                                   name='dilated_au_2_conv1'),
            tf.keras.layers.BatchNormalization(name='normal_au_2_layer'),
            tf.keras.layers.Activation('relu', name='activation_au_2_relu'),
            tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', use_bias=False,
                                   name='dilated_au_2_conv2'),
        ])


        self.softmax = tf.keras.layers.Softmax(name='softmax')


        self.last_conv = tf.keras.layers.Conv2D(filters, 1, name='last_conv')




    '''
    网络训练和运行
    '''
    def call(self, inputs, training=False):
        inputs_shape = tf.shape(inputs)
        B, K, D = inputs_shape[0], inputs_shape[1], inputs_shape[2]
        filters = inputs.get_shape().as_list()[-1]

        x = inputs
        f1 = self.f1_conv(x)
        f2 = self.f2_conv(x)
        f3 = self.f3_conv(x)
        f4 = self.f4_conv(x)

        fb1 = self.add_layer([f1, f2])
        fb2 = self.add_layer([f3, f4])

        wb1 = self.sigmoid_activation(fb1)
        wb2 = self.sigmoid_activation(fb2)

        fr1 = tf.multiply(fb1, wb1)
        fr2 = tf.multiply(fb2, wb2)

        yc = self.pool_layer(fr1)
        zc = self.pool_layer(fr2)

        p1 = self.attention1_conv(yc)
        p2 = self.attention2_conv(zc)

        p1 = self.softmax(p1)
        p2 = self.softmax(p2)

        fa1 = self.add_layer([tf.multiply(f1, p1), tf.multiply(f2, p1)])
        fa2 = self.add_layer([tf.multiply(f3, p2), tf.multiply(f4, p2)])

        fa = self.add_layer([fa1, fa2])

        ffus = self.last_conv(fa)

        return ffus


class ExpNet(tf.keras.Model):
    def __init__(self, num_classes=7, pretrained="msceleb", backbone="resnet50", resnetPooling="avg",feature_dim=512):
        super(ExpNet, self).__init__()
        self.num_classes = num_classes

        self.backbone_type = backbone
        if pretrained is None or pretrained == 'imagenet':
            if backbone=="resnet18":
                from classification_models.tfkeras import Classifiers
                ResNet18, preprocess_input = Classifiers.get('resnet18')
                self.backbone = ResNet18(input_shape=(224,224,3), weights=pretrained, include_top=False, pooling="avg")
            elif backbone=="resnet50":
                self.backbone=ResNet50(input_shape=(112,112,3), weights=pretrained, include_top=False, pooling=resnetPooling)
            elif backbone=="resnet101":
                self.backbone=tf.keras.applications.resnet.ResNet101(input_shape=(224,224,3), weights=pretrained, include_top=False, pooling="avg")
            elif backbone=="resnet152":
                self.backbone = tf.keras.applications.resnet.Resnet152(input_shape=(224, 224, 3), weights=pretrained,
                                                                       include_top=False, pooling="avg")
        elif pretrained=="msceleb":
            if backbone=="resnet18":
                self.backbone = tf.keras.models.load_model("pretrained/resnet18.h5")
                self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
            elif backbone=="resnet50":
                self.backbone = tf.keras.models.load_model("pretrained/resnet50.h5")
            elif backbone=="resnet101":
                self.backbone = tf.keras.models.load_model("pretrained/resnet101.h5")
            elif backbone=="resnet152":
                self.backbone = tf.keras.models.load_model("pretrained/resnet152.h5")


        else:
            raise ValueError('pretrained type invalid, only supports: None, imagenet, and msceleb')
        self.pretrained=pretrained
        self.resnetPooling = resnetPooling
        # ================================================================
        # 增加多通道网络构建的代码
        self.multiChannel = MultiChannel(2048)


        self.pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax',
                                        kernel_initializer=tf.keras.initializers.HeNormal(),
                                        kernel_regularizer=tf.keras.regularizers.L2(0.001)
                                        )
        self.weighting_net = AdaptiveSimNet(feature_dim)

    def call(self, x, training=False):
        if self.pretrained=="msceleb":
            x = tf.transpose(x, (0, 3, 1, 2))
        feat_map = self.backbone(x, training=training)

        if self.resnetPooling == None:
            # 如果resnet不走全连接层，那么就走多通道,此时 feat_map<32,4,4,2048>
            feat_map = self.multiChannel(feat_map)  # todo:先注释掉，看看原来的resnet抽出来全连接层运行如何
            # 多通道走完之后，只是重组了特征，此处需要 全连接层==》softmax层
            feat_map = self.pool(feat_map)


        x = feat_map
        if self.pretrained == "msceleb" and self.backbone_type=='resnet18':
            feat_map = tf.transpose(feat_map, (0, 2, 3, 1))
            x = self.global_pool(feat_map)
        # softmax层
        out = self.fc(x)

        return x, out

def create_model(config):
    model = ExpNet(num_classes=config.num_classes, pretrained=config.pretrained, resnetPooling=config.resnetPooling,backbone=config.backbone)
    model(tf.ones((32, config.input_size[0], config.input_size[1], 3)))
    model.weighting_net(tf.ones((2, config.feature_dim)), tf.ones((2, 4, config.feature_dim)))
    return model


if __name__=="__main__":
    # model = ExpNet(num_classes=7, pretrained="msceleb", backbone="resnet50")
    # model(tf.ones((32, 112, 112, 3)))
    # model.weighting_net(tf.ones((2, 512)), tf.ones((2, 4, 512)))
    # model.summary()

    model = ExpNet(num_classes=7, pretrained="msceleb", backbone="resnet18")
    print(model(tf.ones((32, 224, 224, 3)))[0].shape)
    model.weighting_net(tf.ones((2, 512)), tf.ones((2, 4, 512)))
    model.summary()







