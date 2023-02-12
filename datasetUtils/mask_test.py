import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import shutil, os


def parse_arg(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv_file", type=str,
                        default="/root/autodl-tmp/RAF-DB/data/raf_test.csv",
                        help="")

    parser.add_argument("--gas_csv_file", type=str,
                        default="/root/autodl-tmp/RAF-DB/data/raf_test_gas.csv",
                        help="")
    parser.add_argument("--kn95_csv_file", type=str,
                        default="/root/autodl-tmp/RAF-DB/data/raf_test_kn95.csv",
                        help="")
    parser.add_argument("--surgical_blue_csv_file", type=str,
                        default="/root/autodl-tmp/RAF-DB/data/raf_test_surgical_blue.csv",
                        help="")
    parser.add_argument("--gas_image_dir", type=str,
                        default="/root/autodl-tmp/RAF-DB/Image/mask/gas",
                        help="Path to the directory containing training images")
    parser.add_argument("--kn95_image_dir", type=str,
                        default="/root/autodl-tmp/RAF-DB/Image/mask/kn95",
                        help="Path to the directory containing training images")
    parser.add_argument("--surgical_blue_image_dir", type=str,
                        default="/root/autodl-tmp/RAF-DB/Image/mask/surgical_blue",
                        help="Path to the directory containing training images")
    parser.add_argument("--new_csv_path", type=str,
                        default="/root/autodl-tmp/RAF-DB/data/new_raf_test_mask.csv",
                        help="")
    parser.add_argument("--new_image_path", type=str,
                        default="/root/autodl-tmp/RAF-DB/Image/aligned_mask",
                        help="")


    parser.add_argument("--raw_image_path", type=str,
                        default="/root/autodl-tmp/RAF-DB/Image/aligned",
                        help="Path to the directory containing training images")

    global args
    args = parser.parse_args(argv)


def dealPro(mask_type, raw_image_path, mask_image_dir, file_name, expression):
    # file_name = file_names_gas[i]
    # expression = expression_gas[i]
    if file_path.exists():
        return file_name, expression
    else:
        # 先copy，然后重新命名
        real_file_name = file_name.replace('_' + mask_type, '')
        shutil.copy(raw_image_path + '/' + real_file_name, mask_image_dir + '/' + file_name)
        return real_file_name, expression

if __name__ == '__main__':
    parse_arg()

    df_raw = pd.read_csv(args.raw_csv_file)
    df_gas = pd.read_csv(args.gas_csv_file)
    df_kn95 = pd.read_csv(args.kn95_csv_file)
    df_surgical_blue = pd.read_csv(args.surgical_blue_csv_file)

    file_names_raw = df_raw['subDirectory_filePath'].values
    expression_raw = df_raw['expression'].values
    len_raw = len(file_names_raw)

    file_names_gas = df_gas['subDirectory_filePath'].values
    expression_gas = df_gas['expression'].values
    len_gas = len(file_names_raw)

    file_names_kn95 = df_kn95['subDirectory_filePath'].values
    expression_kn95 = df_kn95['expression'].values
    len_kn95 = len(file_names_raw)

    file_names_surgical_blue = df_surgical_blue['subDirectory_filePath'].values
    expression_surgical_blue = df_surgical_blue['expression'].values
    len_surgical_blue = len(file_names_raw)

    subDirectory_filePath_new = []
    expression_new = []

    # gas_csv_file,kn95_csv_file ,surgical_blue_csv_file ,
    # gas_image_dir ,kn95_image_dir ,surgical_blue_image_dir
    raw_image_path = args.raw_image_path

    for i in range(len_raw):
        swch = i % 3

        if swch == 0 and i < len_gas:
            image_dir = args.gas_image_dir
            file_name = file_names_gas[i]
            expression = expression_gas[i]
            file_path = Path(image_dir, file_name)
            if file_path.exists():
                shutil.copy(file_path, args.new_image_path + '/' + file_name)
            else:
                real_file_name = file_name.replace('_gas', '')
                shutil.copy(raw_image_path + '/' + real_file_name, args.new_image_path + '/' + file_name)

            subDirectory_filePath_new = np.append(subDirectory_filePath_new, file_name)
            expression_new = np.append(expression_new, expression)


        elif swch == 1 and i < len_surgical_blue:
            image_dir = args.surgical_blue_image_dir
            file_name = file_names_surgical_blue[i]
            expression = expression_surgical_blue[i]
            file_path = Path(image_dir, file_name)
            if file_path.exists():
                shutil.copy(file_path, args.new_image_path + '/' + file_name)
            else:
                real_file_name = file_name.replace('_surgical_blue', '')
                shutil.copy(raw_image_path + '/' + real_file_name, args.new_image_path + '/' + file_name)

            subDirectory_filePath_new = np.append(subDirectory_filePath_new, file_name)
            expression_new = np.append(expression_new, expression)

        elif swch == 2 and i < len_kn95:
            image_dir = args.kn95_image_dir
            file_name = file_names_kn95[i]
            expression = expression_kn95[i]
            file_path = Path(image_dir, file_name)
            if file_path.exists():
                shutil.copy(file_path, args.new_image_path + '/' + file_name)
            else:
                real_file_name = file_name.replace('_KN95', '')
                shutil.copy(raw_image_path + '/' + real_file_name, args.new_image_path + '/' + file_name)

            subDirectory_filePath_new = np.append(subDirectory_filePath_new, file_name)
            expression_new = np.append(expression_new, expression)

        else:
            real_file_name = file_names_raw[i]
            expression = expression_raw[i]
            shutil.copy(raw_image_path + '/' + real_file_name, args.new_image_path + '/' + real_file_name)

            subDirectory_filePath_new = np.append(subDirectory_filePath_new, real_file_name)
            expression_new = np.append(expression_new, expression)









    df_new = pd.DataFrame()
    df_new['subDirectory_filePath'] = subDirectory_filePath_new
    expression_new = expression_new.astype(int)
    df_new['expression'] = expression_new

    df_new.to_csv(args.new_csv_path, index=False,
                  header=['subDirectory_filePath', 'expression'])


