# label-distribution-learning-fer-tf
Official Tensorflow Implementation of "Uncertainty-aware Label Distribution Learning for Facial Expression Recognition" paper

# Requirements
python>=3.7

```
pip install tensorflow-gpu==2.4.0
pip install pandas opencv-python scikit-learn matplotlib
pip install git+https://github.com/qubvel/classification_models.git
```

**macos m1**
```shell
# version
numpy=1.22.3
keras=2.11.0
tensorflow-estimator=2.11.0
tensorflow-macos=2.11.0
tensorflow-metal=0.7.0
# pip install
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
pip install numpy --upgrade
```

**autodl服务器**

```shell
conda create -n pytf python=3.8.0
conda init bash
conda activate pytf
pip install tensorflow-gpu==2.4.0
pip install pandas opencv-python scikit-learn matplotlib
pip install git+https://github.com/qubvel/classification_models.git
```


# 外接硬盘文件目录描述
```
autodl-nas
    paper1_noise_result
        best_val_10_8563: 噪音标签为10%训练后的模型，已经损失和准确度记录；
        best_val_20_8455: 噪音标签为20%训练后的模型，已经损失和准确度记录；
        best_val_30_8341: 噪音标签为30%训练后的模型，已经损失和准确度记录；
    paper1_origin_result
        best_val_8732: 原始模型训练的结果
        best_val_8755: 原始模型训练的结果
```


# Datasets
Download the [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset) dataset and extract the `aligned` folder (contains aligned faces) into `data/rafdb/aligned`
We provide the pseudo valence-arousal as well as the pre-built K-nearest-neighbors for each instance in `train.csv`. The annotation file should have the following columns

| subDirectory_filePath | expression | valence | arousal | knn |
|:---------------------:|:----------:|:-------:|:-------:|:---:|
| ...                   | ...        | ...     | ...     | ... |

The preprocessed data annotations are available at [Data Annotation](https://drive.google.com/drive/folders/1nm91FPPj2Se305GWC-zpxqnNfEQvz2uq?usp=sharing)
    
# Trained model
Coming soon...

# Training 
Download our pretrained backbone on MS-Celeb1M and put the .h5 files into `./pretrained` folder: [Google Drive](https://drive.google.com/drive/folders/1xplUqjQ49ozP5VrG3DbzyZkgZ58y80Wo?usp=sharing)


**Run the following command to train the model on datasets**
```Shell
# Training model with resnet50 backbone
python src/train.py --cfg=config_resnet50_raf --train_data_path=data/rafdb/raf_train.csv --train_image_dir=data/rafdb/aligned 

# Training model with resnet18 backbone
python src/train.py --cfg=config_resnet18_raf --train_data_path=data/rafdb/raf_train.csv --train_image_dir=data/rafdb/aligned 

#Resume training from the latest checkpoint
python train.py --train_data=/path/to/train_data.csv --resume
```
**PyCharm run param**
```Shell
--train_data_path /Volumes/out_disk/datasets/RAF-DB/data/raf_train.csv
--train_image_dir /Volumes/out_disk/datasets/RAF-DB/Image/aligned

--val_data_path 
--val_image_dir 
--cfg
--resume


```



# Evaluation
Run this command to evaluate the trained model on the test set. We use the classification accuracy as our evaluation metric.
```Shell

python src/eval.py --cfg=config_resnet50_raf --trained_weights=trained_weights/trained_resnet50_raf --test_data_path=data/rafdb/raf_test.csv --test_image_dir=data/rafdb/aligned

```

-


# linux Run the model

### network log
{'name': 'Base Config', 'backbone': 'resnet50', 'feature_dim': 512, 'pretrained': 'msceleb', 'input_size': [112, 112], 'pad_size': 4, 'batch_size
': 32, 'num_parallel_calls': -1, 'optimizer': 'adam', 'lr': 0.0001, 'lr_decay': 0.1, 'lr_steps': [10, 30], 'gamma': 0.01, 'num_neighbors': 8, 'la
mb_init': 0.5, 'lamb_lr': 10, 'lamb_beta': 0, 'num_classes': 7, 'class_names': ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger'
], 'class_weights': None, 'val_interval': 200, 'save_interval': 2, 'epochs': 60, 'checkpoint_dir': 'weights_checkpoint/resnet50_raf'}

### 服务器上执行代码(aliged)
#### 训练
python src/train.py --cfg=config_resnet50_raf --train_data_path=/home/lz/datasets/RAF-DB/data/raf_train.csv --train_image_dir=/home/lz/datasets/RAF-DB/Image/aligned

python src/train.py --cfg=config_resnet50_raf --train_data_path=/root/autodl-tmp/RAF-DB/data/raf_train.csv --train_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned

python train.py --cfg=config_resnet50_raf --train_data_path=/root/datasets/RAF-DB/data/raf_train.csv --train_image_dir=/root/datasets/RAF-DB/Image/aligned


#### 验证测试
python src/eval.py --cfg=config_resnet50_raf --trained_weights=/home/lz/code/label-distribution-learning-fer-tf/weights_checkpoint/resnet50_raf/epoch_60  --test_data_path=/home/lz/datasets/RAF-DB/data/raf_test.csv --test_image_dir=/home/lz/datasets/RAF-DB/Image/aligned

### 服务器上执行代码(original)
#### 训练
python src/train.py --cfg=config_resnet50_raf --train_data_path=/home/lz/datasets/RAF-DB/data/raf_train_original.csv --train_image_dir=/home/lz/datasets/RAF-DB/Image/original

恢复训练

python src/train.py --cfg=config_resnet50_raf --train_data_path=/home/lz/datasets/RAF-DB/data/raf_train_original.csv --train_image_dir=/home/lz/datasets/RAF-DB/Image/original --resume

#### 验证测试
python src/eval.py --cfg=config_resnet50_raf --trained_weights=/home/lz/code/label-distribution-learning-fer-tf/aligned_wc/resnet50_raf/epoch_60  --test_data_path=/home/lz/datasets/RAF-DB/data/raf_test_original.csv --test_image_dir=/home/lz/datasets/RAF-DB/Image/original

#### aligned测试orignal的数据
python src/eval.py --cfg=config_resnet50_raf --trained_weights=/home/lz/code/label-distribution-learning-fer-tf/aligned_wc/resnet50_raf/epoch_60  --test_data_path=/home/lz/datasets/RAF-DB/data/raf_test_original.csv --test_image_dir=/home/lz/datasets/RAF-DB/Image/original


## AUTODL服务器

### envs
```shell
matplotlib              3.6.2
numpy                   1.23.5
opencv-python           4.6.0.66
pandas                  1.5.2
tensorflow-estimator    2.4.0
tensorflow-gpu          2.4.0

--cfg=config_resnet50_raf
--train_data_path=/root/autodl-tmp/RAF-DB/data/raf_train.csv 
--train_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned

```
# 数据集的处理

## RAF-DB数据集的面部遮挡
**对RAF-DB的测试集图片进行遮挡**
```shell
git clone https://github.com/aqeelanwar/MaskTheFace.git
python mask_the_face.py --path=/root/autodl-tmp/RAF-DB-mask/Image/aligned --mask_type=surgical_blue --color=#ffffff --color_weight=0.5
```
mask_type取值：["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "random", "all"],

color取值：'"#fc1c1a","#177ABC","#94B6D2","#A5AB81","#DD8047","#6b425e","#e26d5a","#c92c48","#6a506d","#ffc900",'
             ' "#ffffff","#000000","#49ff00","#0473e2""'

**此时获取到数据集图像上遮挡，文件的命名也包含遮挡物的类型后缀，所以需要更改csv文件中的文件名**
```shell
cp /root/autodl-tmp/RAF-DB/data/raf_test.csv /root/autodl-tmp/RAF-DB/data/raf_test_gas.csv
// mac命令
sed -i '' "s/.jpg/_gas.jpg/g" /root/autodl-tmp/RAF-DB/data/raf_test_gas.csv
// autodl服务器命令
sed -i "s/.jpg/_gas.jpg/g" /root/autodl-tmp/RAF-DB/data/raf_test_gas.csv

cp /root/autodl-tmp/RAF-DB/data/raf_test.csv /root/autodl-tmp/RAF-DB/data/raf_test_kn95.csv
// mac命令
sed -i '' "s/.jpg/_KN95.jpg/g" /root/autodl-tmp/RAF-DB/data/raf_test_kn95.csv
// autodl服务器命令
sed -i "s/.jpg/_KN95.jpg/g" /root/autodl-tmp/RAF-DB/data/raf_test_kn95.csv


cp /root/autodl-tmp/RAF-DB/data/raf_test.csv /root/autodl-tmp/RAF-DB/data/raf_test_surgical_blue.csv
// mac命令
sed -i '' "s/.jpg/_surgical_blue.jpg/g" /root/autodl-tmp/RAF-DB/data/raf_test_surgical_blue.csv
// autodl服务器命令
sed -i "s/.jpg/_surgical_blue.jpg/g" /root/autodl-tmp/RAF-DB/data/raf_test_surgical_blue.csv
```




## AffectNet数据集的处理
**AffectNet数据集的处理，获取训练接和测试集的csv文件**
```shell
// train
python datasetUtils/affectNet_label.py --image_path=/root/autodl-tmp/AffectNet/train_set/images --label_path=/root/autodl-tmp/AffectNet/train_set/annotations --save_path=/root/autodl-tmp/AffectNet/data/aff_train.csv
// test 
python datasetUtils/affectNet_label.py --image_path=/root/autodl-tmp/AffectNet/val_set/images --label_path=/root/autodl-tmp/AffectNet/val_set/annotations --save_path=/root/autodl-tmp/AffectNet/data/aff_test.csv
```


### RUN the code

**对标论文作者原始代码，使用预训练模型的代码**
```shell
cp /root/autodl-tmp/RAF-DB/pretrained /root/LAL_FER_TF/pretrained
cd /root/LAL_FER_TF
conda activate pytf
git checkout other_code_20230101
python src/train.py --cfg=config_resnet50_raf --train_data_path=/root/autodl-tmp/RAF-DB/data/raf_train.csv --train_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned
python train.py --cfg=config_resnet50_raf --train_data_path=/root/autodl-tmp/RAF-DB/data/raf_train.csv --train_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned
```

**作者论文改造后，使用原始resnet50的代码**
```shell
cd /root/LAL_FER_TF
conda activate pytf
git checkout other_resnet50_train
python train.py --cfg=config_resnet50_raf --train_data_path=/root/autodl-tmp/RAF-DB/data/raf_train.csv --train_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned --pretrained=imagenet
```

**增加多通道网络后的代码**
```shell
cd /root/LAL_FER_TF
conda activate pytf
git checkout other_resnet50_train
python train.py --cfg=config_resnet50_raf --train_data_path=/root/autodl-tmp/RAF-DB/data/raf_train.csv --train_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned --pretrained=imagenet --resnetPooling=None

// 增加保存迭代的参数 && 测试数据集
python train.py --cfg=config_resnet50_raf --train_data_path=/root/autodl-tmp/RAF-DB/data/raf_train.csv --train_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned --pretrained=imagenet --resnetPooling=None --save_interval=2 --val_data_path=/root/autodl-tmp/RAF-DB/data/raf_test.csv --val_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned
// 增加参数验证间隔 【正确的模型训练跑的命令】
python train.py --cfg=config_resnet50_raf --train_data_path=/root/autodl-tmp/RAF-DB/data/raf_train.csv --train_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned --pretrained=imagenet --resnetPooling=None --save_interval=5 --val_interval=200 --val_data_path=/root/autodl-tmp/RAF-DB/data/raf_test.csv --val_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned | tee /root/autodl-nas/paper1_origin_result/train_log.txt

```

# 验证
**一次性测试在多个epoch上的准确率**【废弃】
```shell
python eval.py --cfg=config_resnet50_raf --trained_weights=/root/autodl-tmp/code/LDL_FER_TF/weights_checkpoint/resnet50_raf/epoch_60 --test_data_path=/root/autodl-tmp/RAF-DB/data/raf_test.csv --test_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned --pretrained=imagenet --resnetPooling=None --trained_weights_dir=/root/autodl-tmp/code/LDL_FER_TF/weights_checkpoint/resnet50_raf/
```

## 在pycharm中远程连接autodl，验证模型效果的参数配置

**RAF-DB数据集**
```shell
# 正常的验证逻辑
--cfg=config_resnet50_raf
--trained_weights=/root/autodl-nas/paper1_origin_result/best_val_8755
--test_data_path=/root/autodl-tmp/RAF-DB/data/raf_test.csv
--test_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned
--pretrained=imagenet
--resnetPooling=None
--trained_weights_dir=/root/autodl-tmp/code/LDL_FER_TF/weights_checkpoint/resnet50_raf/

# 10%噪音标签的验证逻辑
--cfg=config_resnet50_raf
--trained_weights=/root/autodl-nas/paper1_noise_result/best_val_10_8563
--test_data_path=/root/autodl-tmp/RAF-DB/data/raf_test.csv
--test_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned
--pretrained=imagenet
--resnetPooling=None
--trained_weights_dir=/root/autodl-tmp/code/LDL_FER_TF/weights_checkpoint/resnet50_raf/

# 20%噪音标签的验证逻辑
--cfg=config_resnet50_raf
--trained_weights=/root/autodl-nas/paper1_noise_result/best_val_20_8455
--test_data_path=/root/autodl-tmp/RAF-DB/data/raf_test.csv
--test_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned
--pretrained=imagenet
--resnetPooling=None
--trained_weights_dir=/root/autodl-tmp/code/LDL_FER_TF/weights_checkpoint/resnet50_raf/


# 30%噪音标签的验证逻辑
--cfg=config_resnet50_raf
--trained_weights=/root/autodl-nas/paper1_noise_result/best_val_30_8341
--test_data_path=/root/autodl-tmp/RAF-DB/data/raf_test.csv
--test_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned
--pretrained=imagenet
--resnetPooling=None
--trained_weights_dir=/root/autodl-tmp/code/LDL_FER_TF/weights_checkpoint/resnet50_raf/


# 面部遮挡的验证逻辑（使用的是正常训练集训练的模型，进行面部遮挡的测试验证）
# 以gas为例
--cfg=config_resnet50_raf
--trained_weights=/root/autodl-nas/paper1_origin_result/best_val_8755
--test_data_path=/root/autodl-tmp/RAF-DB/data/new_raf_test_mask.csv
--test_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned_mask
--pretrained=imagenet
--resnetPooling=None
--trained_weights_dir=/root/autodl-tmp/code/LDL_FER_TF/weights_checkpoint/resnet50_raf/
```



## 噪音标签
**生成不同比例的噪音标签，获取对应的csv文件**
```shell
python datasetUtils/noise_label.py --train_data_path=/root/autodl-tmp/RAF-DB/data/raf_train.csv --dst_data_path=/root/autodl-tmp/RAF-DB/data/ --pre=0.1
```





# 
