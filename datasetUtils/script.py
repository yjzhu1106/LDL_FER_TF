
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



def getRafdbLableDistrubute():
    train_csv = "/root/autodl-tmp/RAF-DB/data/raf_train.csv"
    test_csv = "/root/autodl-tmp/RAF-DB/data/raf_test.csv"
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_exp = train_df['expression'].values
    test_exp = test_df['expression'].values
    expression_dict = {}
    for data in test_exp:
        if data in expression_dict.keys():
            expression_dict[data] = expression_dict[data] + 1
        else:
            expression_dict[data] = 1
    print(expression_dict)

def getAffecnetLabelDistribute():
    train_csv = "/root/autodl-tmp/AffectNet/data/aff_train.csv"
    test_csv = "/root/autodl-tmp/AffectNet/data/aff_test.csv"
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_exp = train_df['expression'].values
    test_exp = test_df['expression'].values
    expression_dict = {}
    for data in test_exp:
        if data in expression_dict.keys():
            expression_dict[data] = expression_dict[data] + 1
        else:
            expression_dict[data] = 1
    print(expression_dict)

def plotAffectNetLabelDistribute():
    train_csv = "/root/autodl-tmp/AffectNet/data/aff_train.csv"
    test_csv = "/root/autodl-tmp/AffectNet/data/aff_test.csv"
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_exp = train_df['expression'].values
    test_exp = test_df['expression'].values
    expression_dict = {}
    for data in train_exp:
        if data in expression_dict.keys():
            expression_dict[data] = expression_dict[data] + 1
        else:
            expression_dict[data] = 1
    print(expression_dict)




    emoLabel = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
    emo_distribute = {}
    for emo_label, emo_label_num in expression_dict.items():
        emo = emoLabel[emo_label]
        #     print(emo)
        emo_distribute[emo] = emo_label_num

    print(emo_distribute)
    # 原始标签分布柱状图
    label_x = []
    label_y = []
    for label, label_num in emo_distribute.items():
        label_x = np.append(label_x, label)
        label_y = np.append(label_y, label_num)
    # Happiness和Anger换位置
    tmp_ax = label_x[1]
    tmp_ay = label_y[1]
    label_x[1] = label_x[3]
    label_y[1] = label_y[3]
    label_x[3] = tmp_ax
    label_y[3] = tmp_ay

    # 正确显示中文和负号
    # plt.rcParams["font.sans-serif"] = ["SimHei"]
    # plt.rcParams["axes.unicode_minus"] = False

    # 画图，plt.bar()可以画柱状图
    for i in range(len(label_x)):
        plt.bar(label_x[i], label_y[i])
    # 设置图片名称
    plt.title("AffectNet label distribute")
    # 设置x轴标签名
    plt.xlabel("lable")
    # 设置y轴标签名
    plt.ylabel("number")
    # 显示
    plt.show()

if __name__ == '__main__':
    train_csv = "/root/autodl-tmp/RAF-DB/data/raf_train.csv"
    test_csv = "/root/autodl-tmp/RAF-DB/data/raf_test.csv"
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_exp = train_df['expression'].values
    test_exp = test_df['expression'].values



