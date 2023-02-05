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
// 增加参数验证间隔
python train.py --cfg=config_resnet50_raf --train_data_path=/root/autodl-tmp/RAF-DB/data/raf_train.csv --train_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned --pretrained=imagenet --resnetPooling=None --save_interval=5 --val_interval=200 --val_data_path=/root/autodl-tmp/RAF-DB/data/raf_test.csv --val_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned | tee /root/autodl-nas/paper1_origin_result/train_log.txt

```



**一次性测试在多个epoch上的准确率**
```shell
python eval.py --cfg=config_resnet50_raf --trained_weights=/root/autodl-tmp/code/LDL_FER_TF/weights_checkpoint/resnet50_raf/epoch_60 --test_data_path=/root/autodl-tmp/RAF-DB/data/raf_test.csv --test_image_dir=/root/autodl-tmp/RAF-DB/Image/aligned --pretrained=imagenet --resnetPooling=None --trained_weights_dir=/root/autodl-tmp/code/LDL_FER_TF/weights_checkpoint/resnet50_raf/
```


**生成噪音标签**
```shell
python datasetUtils/noise_label.py --train_data_path=/root/autodl-tmp/RAF-DB/data/raf_train.csv --dst_data_path=/root/autodl-tmp/RAF-DB/data/ --pre=0.1
```

