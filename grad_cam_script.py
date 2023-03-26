import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
import matplotlib.cm as cm
from model.grad_cam import get_img_array, make_gradcam_heatmap
from tensorflow.python.keras.applications import imagenet_utils
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

from model.models2 import create_model2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Progbar
import pandas as pd
from datetime import datetime

from model.models import create_model
from data_utils import *
import utils

import argparse
import sys
sys.path.append("cfg_files")
sys.path.append("model")

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def parse_arg(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str,
                        default="data/rafdb/raf_train.csv",
                        help="Path to the train_data.csv, should have the following columns:\n'subDirectory_filePath,expression,valence,arousal,knn'",
                        required=True)
    parser.add_argument("--train_image_dir", type=str,
                    default="data/rafdb/aligned",
                    help="Path to the directory containing training images",
                    required=True)
    parser.add_argument("--val_data_path", type=str,
                        default=None,
                        help="Path to the validation_data.csv, should have the following columns:\n'subDirectory_filePath,expression'")
    parser.add_argument("--val_image_dir", type=str,
                    default=None,
                    help="Path to the validation_data.csv, should have the following columns:\n'subDirectory_filePath,expression'")
    # parser.add_argument("--pretrained_weights", type=str,
    #                     default=None,
    #                     help="load the pretrained weights of the model in /path/to/model_weights")
    parser.add_argument("--cfg", type=str,
                        default="config_resnet50_raf",
                        help="config file_name")
    parser.add_argument("--pretrained", type=str,
                        default="msceleb",
                        help="if msceleb, use pretrained model; Or None, use keras.application.resnet50")
    parser.add_argument("--resnetPooling", type=str,
                        default='avg',
                        help="if avg, max, None")
    parser.add_argument("--resume",
                        action= "store_true",
                        help="Resume training from the last checkpoint")

    parser.add_argument("--save_interval", type=int,
                        default=0,
                        help="Resume training from the last checkpoint")

    parser.add_argument("--val_interval", type=int,
                        default=0,
                        help="Resume training from the last checkpoint")

    parser.add_argument("--batch_size", type=int,
                        default=32,
                        help="Resume training from the last checkpoint")



    args = parser.parse_args(argv)
    return args


def img_path(args):
    pass



# 定义Grad-CAM函数
def get_gradcam(model, img_tensor, target_class, layer_name):
    # 计算梯度
    with tf.GradientTape() as tape:
        input_tensor = tf.cast(img_tensor, tf.float32)
        tape.watch(input_tensor)
        outputs = model(input_tensor)
        output = outputs[0][target_class]
    grads = tape.gradient(output, input_tensor)
    # 计算权重
    guided_grads = tf.cast(input_tensor > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    weights = tf.reduce_mean(guided_grads, axis=(1, 2))
    cam = tf.reduce_sum(tf.multiply(weights[:, tf.newaxis, tf.newaxis], model.get_layer(layer_name).output), axis=0)
    cam = cv2.resize(np.float32(cam), (img_tensor.shape[2], img_tensor.shape[1]))
    # 归一化
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    return cam


if __name__ == '__main8__':
    model = ResNet50(input_shape=(112, 112, 3), weights=None, classes=7)
    # 从文件中加载图像，并进行预处理
    img_path = '/root/autodl-tmp/RAF-DB/Image/aligned/train_04591_aligned.jpg'
    img = image.load_img(img_path, target_size=(112, 112))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    # 得到模型的最后一个卷积层
    last_conv_layer = model.get_layer('conv5_block3_out')
    # 使用 GradientTape 计算梯度
    with tf.GradientTape() as tape:
        # 计算模型输出
        preds = model(x)
        # 得到预测类别的梯度
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
        # 计算最后一个卷积层的输出
        last_conv_layer_output = last_conv_layer.output
        # 计算预测类别的梯度相对于最后一个卷积层输出的梯度
        grads = tf.GradientTape(top_class_channel, last_conv_layer_output)
        # 将梯度中的负值变为 0
        grads = tf.where(grads > 0, grads, 0)
        # 计算梯度权重
        weights = tf.reduce_mean(grads, axis=(0, 1))
        # 得到卷积层输出的特征图
        conv_output = last_conv_layer_output[0]
        # 计算类激活热力图
        cam = np.dot(conv_output, weights)
        # 改变热力图尺寸
        cam = tf.image.resize(cam, (112, 112))
        # 归一化热力图
        cam = cam / np.max(cam)




if __name__ == '__main1__':
    # 定义模型
    model = ResNet50(input_shape=(112, 112, 3), weights=None, classes=7)

    # 加载模型权重
    # model.load_weights('resnet50_weights.h5')

    # 加载图片
    img_path = '/root/autodl-tmp/RAF-DB/Image/aligned/train_04591_aligned.jpg'
    img = image.load_img(img_path, target_size=(112, 112))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 得到最后一个卷积层的输出以及分类层的权重
    last_conv_layer = model.get_layer('conv5_block3_out')
    classifier_layer_weights = model.layers[-1].get_weights()[0]

    # 得到预测类别和相应的得分
    preds = model.predict(x)
    preds_tensor = tf.convert_to_tensor(preds)
    predicted_class = np.argmax(preds[0])
    predicted_score = preds[:, predicted_class]



    # 得到预测类别对应的特征图
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = tape.gradient(model.output, [last_conv_layer.output, model.output])
        preds_tensor = preds_tensor[0]

    grads = tape.gradient(preds_tensor, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 得到特征图的权重
    last_conv_layer_output = last_conv_layer_output[0]
    pooled_grads = pooled_grads[0]
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # 将热力图归一化到 0-1 范围内
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # 将热力图叠加到原始图像上
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # 绘制热力图与叠加后的图像
    plt.imshow(heatmap)
    plt.show()

    plt.imshow(superimposed_img)
    plt.show()



def save_and_display_gradcam(img_path, heatmap, cmap="jet", alpha=0.4):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap(cmap)

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose重叠 the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    # superimposed_img.save(cam_path)

    return superimposed_img
    # Display heatmap
    # plt.matshow(superimposed_img)
    # plt.show()

    # Display Grad CAM
    # display(Image(cam_path))


if __name__ == '__main__':
    image_path = '/root/autodl-tmp/RAF-DB/Image/aligned/train_10187_aligned.jpg'
    args = parse_arg()


    config = __import__(args.cfg).config
    if args.pretrained != config.pretrained:
        config.pretrained = args.pretrained
        config.feature_dim = 2048
    if args.resnetPooling == 'None':
        config.resnetPooling = None
    else:
        config.resnetPooling = args.resnetPooling

    if args.save_interval != 0:
        print('Config_save_interval: {}'.format(args.save_interval))
        config.save_interval = args.save_interval

    if args.val_interval != 0:
        print('Config_val_interval: {}'.format(args.val_interval))
        config.val_interval = args.val_interval

    if args.batch_size != 32:
        print('batch_size: {}'.format(args.batch_size))
        config.batch_size = args.batch_size

    print(config.__dict__)

    image = plt.imread(image_path)
    plt.imshow(image)
    plt.show()



    # model = create_model2(config)
    model1 = tf.keras.models.load_model('/root/autodl-tmp/code/LDL_FER_TF/grad_cam/epoch_1.model')  # 加载模型
    model5 = tf.keras.models.load_model('/root/autodl-tmp/code/LDL_FER_TF/grad_cam/epoch_5.model')  # 加载模型
    # model10 = tf.keras.models.load_model('/root/autodl-tmp/code/LDL_FER_TF/grad_cam/epoch_10.model')  # 加载模型
    model15 = tf.keras.models.load_model('/root/autodl-tmp/code/LDL_FER_TF/grad_cam/epoch_15.model')  # 加载模型
    model20 = tf.keras.models.load_model('/root/autodl-tmp/code/LDL_FER_TF/grad_cam/epoch_20.model')  # 加载模型

    print("Model created!")

    img_size = (112, 112)
    preprocess_input = imagenet_utils.preprocess_input
    decode_predictions = imagenet_utils.decode_predictions

    last_conv_layer_name = "multiChannel"

    img_array = preprocess_input(get_img_array(image_path, size=img_size))

    # Remove last layer's softmax
    # model.layers[-1].activation = None

    # Print what the top predicted class is
    feat_preds1, preds1, multi_fea1 = model1(img_array)
    feat_preds5, preds5, multi_fea5 = model5(img_array)
    # feat_preds10, preds10, multi_fea10 = model10(img_array)
    feat_preds15, preds15, multi_fea15 = model15(img_array)
    feat_preds20, preds20, multi_fea20 = model20(img_array)
    # print("Predicted:", decode_predictions(fea_preds, top=1)[0])  # 这些地方所加的0皆是batch中第一个sample的意思
    print("epoch:1, Predicted:", np.argmax(preds1))
    print("epoch:5, Predicted:", np.argmax(preds5))
    # print("epoch:10, Predicted:", np.argmax(preds10))
    print("epoch:15, Predicted:", np.argmax(preds15))
    print("epoch:20, Predicted:", np.argmax(preds20))

    # Generate class activation heatmap
    heatmap1 = make_gradcam_heatmap(img_array, model1, last_conv_layer_name)
    heatmap5 = make_gradcam_heatmap(img_array, model5, last_conv_layer_name)
    # heatmap10 = make_gradcam_heatmap(img_array, model10, last_conv_layer_name)
    heatmap15 = make_gradcam_heatmap(img_array, model15, last_conv_layer_name)
    heatmap20 = make_gradcam_heatmap(img_array, model20, last_conv_layer_name)

    # Display heatmap
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    axs[0].matshow(heatmap1)
    axs[1].matshow(heatmap5)
    # axs[2].matshow(heatmap10)
    axs[3].matshow(heatmap15)
    axs[4].matshow(heatmap20)
    plt.show()


    superimposed_img1 = save_and_display_gradcam(image_path, heatmap1,cmap="jet", alpha=0.4)
    superimposed_img5 = save_and_display_gradcam(image_path, heatmap5)
    # superimposed_img10 = save_and_display_gradcam(image_path, heatmap10)
    superimposed_img15 = save_and_display_gradcam(image_path, heatmap15)
    superimposed_img20 = save_and_display_gradcam(image_path, heatmap20)

    fig, axs = plt.subplots(1, 10, figsize=(10, 2))
    for i in range(10):
        alpha = i * 0.1
        su = save_and_display_gradcam(image_path, heatmap1, alpha=alpha)
        axs[0].matshow(su)
    plt.show()


    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    axs[0].matshow(superimposed_img1)
    axs[1].matshow(superimposed_img5)
    # axs[2].matshow(superimposed_img10)
    axs[3].matshow(superimposed_img15)
    axs[4].matshow(superimposed_img20)
    plt.show()



