import numpy as np
import os
from pathlib import Path
import pandas as pd
import argparse
import math
import heapq
import cupy as cp
import time
from tensorflow.keras.utils import Progbar
import mars
import mars.remote as mr
import mars.tensor as mt


def parse_arg(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str,
                        default="/root/autodl-tmp/AffectNet/val_set/images",
                        help="affectNet数据集图片的路径",
                        required=True)
    parser.add_argument("--label_path", type=str,
                        default="/root/autodl-tmp/AffectNet/val_set/annotations",
                        help="affectNet数据集标签和VA的路径")
    parser.add_argument("--save_path", type=str,
                        default="/root/autodl-tmp/AffectNet/data/aff_test.csv",
                        help="affectNet数据集标签和VA的路径")
    args = parser.parse_args(argv)
    return args


def getAttr(image_path, label_path):
    subDirectory_filePath = []
    expression = []
    valence = []
    arousal = []

    expression_backup = []
    pb_i = Progbar(287651, width=50, interval=0.01,
                   stateful_metrics=['_exp', '_aro', '_val'])

    i = 1
    for image_file_path in image_path.iterdir():
        # if i == 200:
        #     break
        i = i+ 1

        flag_exp = 0
        flag_aro = 0
        flag_val = 0

        image_name = image_file_path.name.split('.')[0]

        img_label_name = image_name + '_exp.npy'  # 表情标签的文件名
        img_aro_name = image_name + '_aro.npy'  # 表情VA值的A、
        img_val_name = image_name + '_val.npy'  # 表情VA值的V

        img_label_path = Path(label_path, img_label_name)
        img_aro_path = Path(label_path, img_aro_name)
        img_val_path = Path(label_path, img_val_name)

        if img_label_path.exists() and not img_label_path.is_dir():
            label = int(np.load(img_label_path).astype(int))
            flag_exp = 1
        else:
            print(img_label_name + ' not exists!!!')

        if img_aro_path.exists() and not img_aro_path.is_dir():
            aro = float(np.load(img_aro_path).astype(float))
            flag_aro = 1
        else:
            print(img_aro_name + ' not exists!!!')

        if img_val_path.exists() and not img_val_path.is_dir():
            val = float(np.load(img_val_path).astype(float))
            flag_val = 1
        else:
            print(img_val_name + ' not exists!!!')

        subDirectory_filePath = np.append(subDirectory_filePath, image_file_path.name)
        expression = np.append(expression, label)
        valence = np.append(valence, aro)
        arousal = np.append(arousal, val)

        expression_backup = np.append(expression_backup, label)
        pb_i.add(1, [('_exp', flag_exp),
                     ('_aro', flag_aro),
                     ('_val', flag_val)])

    expression = expression.astype(int)

    return subDirectory_filePath, expression, valence, arousal


def getKnn(subDirectory_filePath, valence, arousal):


    samples = len(subDirectory_filePath)
    pb_j = Progbar(samples, width=50, interval=0.01,
                   stateful_metrics=['knn','time'])


    print('>>>>>>>>>>>>start computer distance>>>>>>>>>>>>>')
    start = time.time()
    knn = []  # 二维数组，第一维对应每一个图片，第二位是20个最近的邻居；

    save_interval = 20
    flag = 0
    for i in range(samples):
        flag = flag + 1
        # k_neighbors = []
        # root_v = mt.tensor(valence[i], gpu=True)
        # root_a = mt.tensor(arousal[i], gpu=True)
        root_v = valence[i]
        root_a = arousal[i]
        s = time.time()
        k_neighbors = [float(get_distance(valence, arousal, root_a, root_v, j)) for j in range(samples)]
        e = time.time()
        # print('i: {}, time: {}'.format(i, (e-s)))
        knn.append(k_neighbors)
        pb_j.add(1, [('knn', len(knn)),('time', (e-s))])
        if i % save_interval == 0:
            np.save("/root/autodl-tmp/AffectNet/data/knn_distance.npy", knn)

        # k_neighbors = [mr.spawn(get_distance, args=(valence, arousal, root_a, root_v, j))
        #                 for j in range(samples)]


        # for j in range(samples):
        #     node_v = valence[j]
        #     node_a = arousal[j]
        #
        #     diff_a = cp.subtract(node_a, root_a)
        #     diff_b = cp.subtract(node_v, root_v)
        #
        #     distance = cp.sqrt(cp.add(cp.multiply(diff_a, diff_a),cp.multiply(diff_b, diff_b)))
        #     k_neighbors = np.append(k_neighbors, distance)
        # result = getMinIndex(cp.asnumpy(k_neighbors), i)
        # knn.append(k_neighbors)

        # knn = mr.spawn(getMinIndex, args=(k_neighbors, i))
        # result = knn.execute()
        # knn.append()
    end = time.time()
    print('>>>>>>>>>>>>end computer distance, time:  %s Seconds>>>>>>>>>>>>' % (end - start))
    result = [getMinIndex(knn[i], i) for i in range(len(knn))]
    end2 = time.time()
    print('>>>>>>>>>>>>end computer distance min index, time:  %s Seconds>>>>>>>>>>>>' % (end2 - end))

    return result


def get_distance(valence, arousal, root_a, root_v, j):
    node_v = valence[j]
    node_a = arousal[j]

    diff_a = cp.subtract(node_a, root_a)
    diff_b = cp.subtract(node_v, root_v)

    distance = cp.sqrt(cp.add(cp.multiply(diff_a, diff_a), cp.multiply(diff_b, diff_b)))
    # diff_a = node_a - root_a
    # diff_v = node_v - root_v
    # distance = math.sqrt(diff_v * diff_v + diff_a * diff_a)

    return distance


def getMinIndex(arr, root_index):
    # arr = cp.asnumpy(arr)
    arr = list(arr)
    ##求最小的5个值
    arr_min = heapq.nsmallest(21, arr)  # 获取最小的五个值并排序
    index_min = map(arr.index, arr_min)  # 获取最小的五个值的下标
    # print(arr_min)

    index_min = list(index_min)
    # map生成的对象要转化成为list才能输出
    dis_index_dict = {}
    for i in range(len(arr_min)):
        dis_index_dict[arr_min[i]] = index_min[i]

    arr_min.sort()

    result = ''
    for dis in arr_min:
        index = dis_index_dict[dis]
        if root_index == index:
            continue
        result = result + str(index) + ';'
    result = result.strip(';')

    return result


if __name__ == '__main__':
    # mars.new_session()
    args = parse_arg()

    image_path = Path(args.image_path)
    label_path = Path(args.label_path)
    print('>>>>>>>beigin get image attribute>>>>>>>>>>>>')
    print()
    start = time.time()
    subDirectory_filePath, expression, valence, arousal = getAttr(image_path=image_path, label_path=label_path)
    end = time.time()
    print()
    print('>>>>>>>end get image attribute, time:  %s Seconds>>>>>>>>>>>>' % (end - start))

    df = pd.DataFrame(subDirectory_filePath)
    # df.columns = ['subDirectory_filePath', 'expression', 'valence', 'arousal', 'knn', 'expression_backup']
    df.columns = ['subDirectory_filePath']
    df['expression'] = expression
    df['valence'] = valence
    df['arousal'] = arousal

    np.save("/root/autodl-tmp/AffectNet/data/expression.npy", expression)
    np.save("/root/autodl-tmp/AffectNet/data/valence.npy", valence)
    np.save("/root/autodl-tmp/AffectNet/data/arousal.npy", arousal)

    '''转换数组为GPU可以接受的数组，来用GPU计算'''
    v_cp = cp.asarray(valence)
    a_cp = cp.asarray(arousal)

    # v_cp = mt.tensor(valence, gpu=True)
    # a_cp = mt.tensor(arousal, gpu=True)
    '''获取每个图片最近的20个图片'''
    knn = getKnn(subDirectory_filePath, v_cp, a_cp)
    df['knn']=knn

    df.to_csv(args.save_path, index=False,
              header=['subDirectory_filePath', 'expression', 'valence', 'arousal', 'knn'])

    mars.stop_server()
