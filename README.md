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

