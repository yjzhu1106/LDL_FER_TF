import pandas as pd
import collections
import copy
import warnings
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib
import argparse
np.set_printoptions(suppress=True)

def parse_arg(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str,
                        default="/root/autodl-tmp/RAF-DB/data/raf_train.csv",
                        help="原始数据集标签的路径",
                        required=True)
    parser.add_argument("--dst_data_path", type=str,
                        default="/root/autodl-tmp/RAF-DB/data/",
                        help="噪音标签处理后，保存的路径, 名字需要自己定义")
    parser.add_argument("--pre", type=float,
                        default=0.1,
                        help="噪音标签占比")
    args = parser.parse_args(argv)
    return args


def recreat_data(name, raw_path, dest_path, pre):
    df = pd.read_csv(raw_path, header=None)
    data = df.values
    numSamples, _ = data.shape
    samNums = (int)(numSamples * pre)  # 需要置换标签总的样本数量
    k_list = list(collections.Counter(data[1:, 1]).keys())  # 读取样本点类别
    v_list = list(collections.Counter(data[1:, 1]).values())  # 样本点类别对应的样本数目
    print(k_list)
    dff = {}
    for i in range(len(k_list)):  # 类别循环
        tag = k_list[i]  # 标签类别
        dff[i] = df[df[0] == int(tag)].reset_index()  # 提取该类别的样本点
        samNumsi = int(samNums * (v_list[i] / numSamples))  # 该类别中需要置换的样本数目
        temp_k = copy.deepcopy(k_list)
        temp_v = copy.deepcopy(v_list)
        temp_k.pop(i)  # 将该类别出栈，只保留剩余的类别
        temp_v.pop(i)  # 将该类别出栈，只保留剩余的类别的样本点数目
        print("tempk", temp_k)
        print("temp_v", temp_v)
        k = 0
        samNumsij = 0
        for j in range(len(temp_k)):
            samNumsij += int(samNumsi * (temp_v[j] / (sum(temp_v))))  # 剩余类别需要置换的样本点数目
            print("sam", samNumsij)
            for l in range(k, samNumsij):
                dff[i].loc[l, 0] = int(temp_k[j])
                k += 1
    new = dff[0]
    for i in range(len(k_list) - 1):
        new = pd.concat([new, dff[i + 1]])
    new = shuffle(new).reset_index().drop(['index', 'level_0'], axis=1)  # 打乱顺序
    new.to_csv(dest_path + name, index=False, header=None)


def recreat_data2(name, raw_path, dest_path, pre):
    df = pd.read_csv(raw_path)
    df['expression_backup'] = df['expression']
    data = df.values
    numSamples, _ = data.shape
    label_index_dict = {}  # 将每一个标签的索引进行存储，方便后续重新定义标签

    '''处理之后，获得每一个表情分类，相关的下标'''
    for index, sample in enumerate(data):
        index_arr = []
        if sample[1] in label_index_dict.keys():  # sample[1]是标签值， sample是一条记录的数据，最后只改变标签值即可
            index_arr = label_index_dict[sample[1]]
        index_arr = np.append(index_arr, index)  # 将当前的标签添加进去
        label_index_dict[sample[1]] = index_arr

    '''计算每一个表情标签，应该有多少个转换成其他的标签，并且其他标签分别应该转换多少个'''
    k_list = list(collections.Counter(data[1:, 1]).keys())  # 读取样本点类别
    v_list = list(collections.Counter(data[1:, 1]).values())  # 样本点类别对应的样本数目

    label_convert = {}
    for i in range(len(k_list)):
        label = k_list[i]
        label_num = v_list[i]
        convert_num = int(label_num * pre)
        # 将现在的类别出栈，然后遍历，看其他标签应该转化多少个
        temp_k = copy.deepcopy(k_list)
        temp_v = copy.deepcopy(v_list)
        temp_k.pop(i)
        temp_v.pop(i)

        # 当前label的所有索引
        indexs = label_index_dict[label]
        np.random.shuffle(indexs)

        for j in range(len(temp_k)):
            other_label = temp_k[j]
            other_label_num = temp_v[j]
            convert_other_label_num = int(convert_num * (other_label_num / (sum(temp_v))))
            convert_index = indexs[:convert_other_label_num]

            print('label: {}, convert_num: {}, indexLen:{}, to: {}-{}'.format(
                label, convert_num, len(indexs), other_label, convert_other_label_num))

            for ind in convert_index:
                label_convert[int(ind)] = other_label
            indexs = np.delete(indexs, range(convert_other_label_num))

    # print(label_convert)
    '''找到了需要进行转换的索引，然后将这些索引对应的数据，转换成需要的数据。'''
    for index, convert_label in label_convert.items():
        data[index][1] = convert_label
    new_pd = pd.DataFrame(data)
    new_pd.to_csv(dest_path+name, index=False, header=None)


if __name__ == '__main__':
    args = parse_arg()

    train_data_path = args.train_data_path
    name = train_data_path.split('/')[-1].split('.')[0] + '_' + str(int(args.pre * 100)) + '.csv'

    recreat_data2(name=name,
                  raw_path=train_data_path,
                  dest_path=args.dst_data_path,
                  pre=args.pre)
