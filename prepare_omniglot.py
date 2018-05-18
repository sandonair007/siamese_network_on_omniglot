import os
import numpy as np
from scipy.misc import imread

import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Dataset')
    parser.add_argument('--data-path', dest='data_path', help='path to omniglot folder',
                        default=os.path.join(os.getcwd(), 'omniglot'), type=str)
    parser.add_argument('--save-path', dest='save_path', help='path to generated pickle files',
                        default=os.path.join(os.getcwd(), 'omniglot'), type=str)
    args = parser.parse_args()
    return args

def load_images(path, data_file, enable_wr=True):
    # unzip the file if data is not unzipped
    if not os.path.exists(path):
        print("unzipping --------->")
        os.chdir(data_path)
        os.system("unzip {}".format(path+".zip"))

    # stack images of same label
    alphas = os.listdir(path)
    lang_dict = {}
    char_dict = {}
    all_images = {}

    cur_label = 0
    img_index = 0
    for alpha in alphas:
        print("loading alphabet: " + alpha)
        lang_dict[alpha] = [cur_label, None]
        alpha_path = os.path.join(path,alpha)
        for letter in os.listdir(alpha_path):
            char_dict[cur_label] = (alpha, letter)
            img_list = []
            letter_path = os.path.join(alpha_path, letter)
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                img = imread(image_path)
                img_list.append(img)
                img_index += 1
            all_images[cur_label] = np.stack(img_list)
            cur_label += 1

        lang_dict[alpha][1] = cur_label - 1

    if enable_wr:
        with open(data_file, "wb") as f:
            print("pickling to file ---------->")
            pickle.dump((all_images, char_dict), f)
            print("saved to file: " + data_file)

    return lang_dict, char_dict


if __name__ == '__main__':
    args = parse_args()

    data_path = os.path.join(args.data_path,"python")

    train_folder = os.path.join(data_path,"images_background")
    val_folder = os.path.join(data_path,"images_evaluation")

    train_data = os.path.join(args.save_path, "train.pickle")
    val_data = os.path.join(args.save_path, "val.pickle")

    tr_lang, tr_char= load_images(train_folder, train_data, enable_wr=True)
    va_lang, va_char= load_images(val_folder, val_data, enable_wr=True)

