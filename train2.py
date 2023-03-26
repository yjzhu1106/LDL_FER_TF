import os

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

    parser.add_argument("--num_neighbors", type=int,
                        default=8,
                        help='the number of neighbors')



    args = parser.parse_args(argv)
    return args

def train_step(model,model_adpt, optimizer, x_batch_train, y_batch_train, x_batch_aux, config,
               global_labels, lamb_hat, lamb_optim, dcm_loss=None,
               loss_weights_dict=None, class_weights_dict=None,
               knn_weights=None,
               lamb=None, idx=None, neighbor_idx=None):
    if loss_weights_dict is None:
        loss_weights_dict = {
            'emotion_loss': 1,
            'va_loss': 1,
        }

    sample_weights = utils.create_sample_weights(y_batch_train[0], class_weights_dict)
    # Get neighbor prediction

    B, K, H, W, _ = x_batch_aux.shape
    x_batch_aux = tf.reshape(x_batch_aux, shape=(-1, H, W, 3))
    feat_aux, preds_aux, multi_feat_aux = model(x_batch_aux, training=True)



    preds_aux = tf.reshape(preds_aux, shape=(B, K, -1))  # shape (B,K,C)
    feat_aux = tf.reshape(feat_aux, shape=(B, K, -1))  # shape (B,K,C)

    with tf.GradientTape() as tape:
        feat, preds, multi_feat = model(x_batch_train, training=True)
        attention_weights = model_adpt((feat), (feat_aux), training=True)
        # attention_weights = model.weighting_net((feat), (feat_aux), training=True)

        # construct label distribution
        emotion_cls_pred = preds
        emotion_cls_true = y_batch_train[0]


        if idx is not None:
            lamb = tf.gather(lamb_hat, idx)
            lamb = tf.sigmoid(lamb)

        emotion_cls_true = utils.construct_target_distribution(emotion_cls_true, (preds_aux), knn_weights,
                                                               (attention_weights), lamb=lamb)

        emotion_loss = utils.CELoss(emotion_cls_true, emotion_cls_pred, sample_weights)

        dcm_loss_value = dcm_loss(feat, y_batch_train[0], feat_aux, tf.gather(global_labels, neighbor_idx),
                                    lamb_hat=lamb_hat,
                                    indices = tf.concat([idx, tf.reshape(neighbor_idx, -1)], axis=0))

        total_loss = emotion_loss + config.gamma * dcm_loss_value

        optimizing_variables = model.trainable_variables + [lamb_hat]
        gradients = tape.gradient(total_loss, optimizing_variables)
        gradients, lamb_hat_grad = gradients[:-1], gradients[-1]

        optimizer.apply_gradients(
            [(grad, var) for (grad, var) in zip(gradients, optimizing_variables) if grad is not None])

        lamb_optim.apply_gradients([(lamb_hat_grad, lamb_hat)])

        dcm_loss.update_centers(dcm_loss.compute_grads())

    return preds, total_loss, {'emotion_loss': emotion_loss,
                               'lamb': tf.reduce_mean(tf.reduce_mean(tf.sigmoid(tf.gather(lamb_hat, idx))))}


def test_step(model, x_batch, y_batch):
    feat, preds, multi_feat = model(x_batch, training=False)

    emotion_cls_pred = preds
    emotion_cls_true = y_batch[0]
    emotion_loss = tf.keras.losses.SparseCategoricalCrossentropy()(emotion_cls_true, emotion_cls_pred)

    total_loss = emotion_loss

    return preds, total_loss, {'emotion_loss': emotion_loss}






def train(model,model_adpt, optimizer, train_dataset, global_labels, config,
          val_dataset=None, epochs=5, load_checkpoint=False,
          loss_weights_dict=None,
          class_weights=None):
    # define metrics for controlling process
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy')

    batches_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    ckpt_dir = config.checkpoint_dir

    # init values
    dcm_loss = utils.DiscriminativeLoss(config.num_classes, feature_dim=config.feature_dim)
    lamb_hat = tf.Variable(tf.zeros(shape=(len(global_labels), 1), dtype=tf.float32) + config.lamb_init)
    lamb_optim = tf.keras.optimizers.SGD(config.lamb_lr, config.lamb_beta)
    class_weights_dict = None
    if class_weights:
        if type(class_weights) == bool:
            class_weights = sklearn.utils.class_weight.compute_class_weight("balanced",
                                                                            classes=np.unique(global_labels.numpy()),
                                                                            y=global_labels.numpy())
            print(class_weights)
        class_weights_dict = {i:v for i,v in enumerate(class_weights)}
    best_val = 0
    iter_count = 0
    val_interval = config.val_interval
    save_interval = config.save_interval



    # setup checkpoint manager
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model, optimizer=optimizer,
                                     dcm_loss=dcm_loss)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="./checkpoints", max_to_keep=1
    )
    if load_checkpoint:
        status = checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from sratch....")
    else:
        print("Initializing from scratch...")

    iter_count = checkpoint.step.numpy()

    # todo: 记录迭代次数下，损失和准确度
    df = pd.DataFrame(columns=['time', 'epoch', 'loss', 'accuracy', 'lamb', 'val_loss', 'val_accuracy'])  # 列名
    df.to_csv(f'{ckpt_dir}/train_log.csv', index=False)  # 路径可以根据需要更改
    save_model = 1
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        print("Epoch {}".format(int(iter_count / batches_per_epoch) + 1))
        print("LR: ", float(optimizer.learning_rate(optimizer.iterations)))

        # Training
        pb_i = Progbar(batches_per_epoch, width=30, interval=0.5,
                       stateful_metrics=['total_loss', 'emotion_loss', 'emotion_acc', 'avg_lamb'])
        save_lamb = 0
        for x_batch_train, va_regression_true, emotion_cls_true, x_batch_aux, knn_weights, idx, neighbor_idx in train_dataset:
            checkpoint.step.assign_add(1)
            iter_count += 1
            curr_epoch = int(iter_count / batches_per_epoch)

            lamb = None #we use trainable lambda
            y_batch_train = (emotion_cls_true, va_regression_true)
            preds, total_loss, loss_dict = train_step(model, model_adpt, optimizer, x_batch_train, y_batch_train, x_batch_aux, config,
                                                      global_labels, lamb_hat, lamb_optim, dcm_loss=dcm_loss,
                                                      class_weights_dict=class_weights_dict,
                                                      knn_weights=knn_weights,
                                                      lamb=lamb, idx=idx, neighbor_idx=neighbor_idx,
                                                      )
            train_loss(total_loss)
            emotion_acc = train_accuracy(emotion_cls_true, preds)

            pb_i.add(1, [('total_loss', total_loss.numpy()),
                         ('emotion_accuracy', emotion_acc),
                         ('avg_lamb', loss_dict['lamb'])])

            save_lamb = loss_dict['lamb']

            if iter_count % val_interval == 0 and val_dataset is not None:
                val_loss.reset_states()
                val_accuracy.reset_states()
                for x_batch, va_regression_true, emotion_cls_true in val_dataset:
                    y_batch = (emotion_cls_true, va_regression_true)
                    preds, total_loss, loss_dict = test_step(model, x_batch, y_batch)
                    val_loss(total_loss)  # update metric
                    val_accuracy(emotion_cls_true, preds)  # update metric
                acc = val_accuracy.result()
                print("\n---Iterations: {}, Val loss: {:.4}, Val Acc: {:.4}".format(iter_count,val_loss.result(), acc))
                if (acc > best_val):
                    model.save_weights(f"{ckpt_dir}/best_val/Model")
                    print("====Best validation model saved!====")
                    best_val = acc

        save_path = manager.save()
        if (curr_epoch) % save_interval == 0:
            model.save_weights(f'{ckpt_dir}/epoch_' + str(curr_epoch) + '/Model')

        print('End of Epoch: {}, Iter: {}, Train Loss: {:.4}, Emotion Acc: {:.4}'.format(curr_epoch, iter_count,
                                                                                         train_loss.result(),
                                                                                         train_accuracy.result()))


        # Validation
        if val_dataset is not None:
            val_loss.reset_states()
            val_accuracy.reset_states()

            for x_batch, va_regression_true, emotion_cls_true in val_dataset:
                y_batch = (emotion_cls_true, va_regression_true)
                preds, total_loss, loss_dict = test_step(model, x_batch, y_batch)

                val_loss(total_loss)  # update metric
                val_accuracy(emotion_cls_true, preds)  # update metric

            print('Val loss: {:.4},  Val accuracy: {:.4}'.format(val_loss.result(), val_accuracy.result()))
            print('===================================================')

            if (val_accuracy.result() > best_val):
                model.save_weights(f"{ckpt_dir}/best_val/Model")
                print("====Best validation model saved!====")
                best_val = val_accuracy.result()
        print()

        # todo: 记录每次迭代的损失和准确度
        list = ["%s" % datetime.now(), epoch,
                '%.4f' % train_loss.result(),
                '%.4f' % train_accuracy.result(),
                '%.4f' % save_lamb,
                '%.4f' % val_loss.result(),
                '%.4f' % val_accuracy.result(),
                ]
        data = pd.DataFrame([list])
        data.to_csv(f'{ckpt_dir}/train_log.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

        if save_model == 1:
            model.save('/root/autodl-tmp/code/LDL_FER_TF/grad_cam/epoch_1.model')

        if save_model == 5:
            model.save('/root/autodl-tmp/code/LDL_FER_TF/grad_cam/epoch_5.model')

        if save_model == 10:
            model.save('/root/autodl-tmp/code/LDL_FER_TF/grad_cam/epoch_10.model')

        if save_model == 15:
            model.save('/root/autodl-tmp/code/LDL_FER_TF/grad_cam/epoch_15.model')

        if save_model == 20:
            model.save('/root/autodl-tmp/code/LDL_FER_TF/grad_cam/epoch_20.model')

        save_model = save_model + 1

    return model


def main(train_data_path, image_dir, config, val_data_path=None, val_image_dir = None):
    assert (val_image_dir is not None) or (val_data_path is None)

    model, model_adpt = create_model2(config)
    print("Model created!")

    train_dataset = get_train_dataset(train_data_path, image_dir, config)
    optimizer = utils.get_optimizer(train_dataset, config)
    val_dataset = get_test_dataset(val_data_path, val_image_dir, config) if val_data_path is not None else None
    print("Dataset loaded!")

    train_data = pd.read_csv(train_data_path)
    global_labels = tf.constant(train_data['expression'], dtype=tf.int32)


    print("Start training...")
    train(model, model_adpt,optimizer, train_dataset, global_labels, config,
          val_dataset=val_dataset,
          epochs=config.epochs,
          load_checkpoint=args.resume,
          class_weights=config.class_weights)

if __name__ == '__main__':
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

    if args.num_neighbors != 8:
        print('num_neighbors: {}'.format(args.num_neighbors))
        config.num_neighbors = args.num_neighbors

    print(config.__dict__)

    main(train_data_path= args.train_data_path, image_dir=args.train_image_dir, config= config,
         val_data_path=args.val_data_path, val_image_dir=args.val_image_dir)

