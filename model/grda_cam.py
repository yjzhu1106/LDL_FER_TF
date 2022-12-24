

import pandas as pd
import numpy as np
import cv2
from keras import backend as K
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import image_utils
# image_utils.array_to_img
# image_utils.img_to_array
# image_utils.load_img
from keras.models import load_model


K.set_learning_phase(1) #set learning phase

def Grad_Cam(model, x, layer_name):

    # 前処理
    X = np.expand_dims(x, axis=0)

    X = X.astype('float32')
    preprocessed_input = X / 255.0


    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]


    conv_output = model.get_layer(layer_name).output   # layer_name
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables)
    gradient_function = K.function([model.input], [conv_output, grads])  # model.input

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]


    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)




    cam = cv2.resize(cam, (200, 200), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)
    jetcam = (np.float32(jetcam) + x / 2)

    return jetcam




