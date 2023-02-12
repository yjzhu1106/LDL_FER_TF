import pandas as pd
import  numpy as np

if __name__ == '__main__':
    backup1_path = '/root/autodl-nas/paper1_noise_result/best_val_20_8556/train_log_backup.csv'
    backup2_path = '/root/autodl-nas/paper1_noise_result/best_val_20_8556/train_log_backup_2.csv'
    dst_path = '/root/autodl-nas/paper1_noise_result/best_val_20_8556/train_log_backup_2.csv'


    df1 = pd.read_csv(backup1_path)
    df2 = pd.read_csv(backup2_path)

    time1 = df1['time'].values
    epoch1 = df1['epoch'].values
    loss1 = df1['loss'].values
    accuracy1 = df1['accuracy'].values
    lamb1 = df1['lamb'].values
    val_loss1 = df1['val_loss'].values
    val_accuracy1 = df1['val_accuracy'].values

    time2 = df2['time'].values
    epoch2 = df2['epoch'].values
    loss2 = df2['loss'].values
    accuracy2 = df2['accuracy'].values
    lamb2 = df2['lamb'].values
    val_loss2 = df2['val_loss'].values
    val_accuracy2 = df2['val_accuracy'].values

    check_point = len(time1)
    for i in range(len(time2)):


        time1 = np.append(time1, time2[i])
        epoch1 = np.append(epoch1, check_point)
        loss1 = np.append(loss1, loss2[i])
        accuracy1 = np.append(accuracy1, accuracy2[i])
        lamb1 = np.append(lamb1, lamb2[i])
        val_loss1 = np.append(val_loss1, val_loss2[i])
        val_accuracy1 = np.append(val_accuracy1, val_accuracy2[i])

        check_point = check_point + 1

    df_new = pd.DataFrame()
    df_new['time'] = time1
    df_new['epoch'] = epoch1
    df_new['loss'] = loss1
    df_new['accuracy'] = accuracy1
    df_new['lamb'] = lamb1
    df_new['val_loss'] = val_loss1
    df_new['val_accuracy'] = val_accuracy1
# time,epoch,loss,accuracy,lamb,val_loss,val_accuracy

    df_new.to_csv(dst_path,  index=False,
                  header=['time','epoch','loss','accuracy','lamb','val_loss','val_accuracy'])










