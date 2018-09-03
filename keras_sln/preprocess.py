import os
import glob
import cv2 as cv
import skimage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K


class DataProcess(object):
    def __init__(self):
        self.file_path = os.path.dirname(os.path.abspath(__file__))  # file path of this project
        self.data_path = os.path.split(self.file_path)[0] + os.sep + 'images' + os.sep  # dir of images and labels
        self.image_list = []  # list of images [image_name, image_data]
        self.im_rows, self.im_cols, self.channels = 326, 490, 3  # the size of images (without considering channel)

    # import images and labels
    def import_images(self):
        if os.path.exists(self.data_path):
            print("file path exist")
            files = glob.glob(self.data_path + '*.jpg')
            for image in files:
                im = cv.imread(image)
                im_name = image.split(os.sep)[-1]
                im = cv.resize(im, (self.im_cols, self.im_rows), interpolation=cv.INTER_CUBIC)
                # turn image from uint8 to float64, make range into [0, 1], normalization(worse for this project)
                # im = skimage.img_as_float(im)
                self.image_list.append([im_name, im])
        else:
            raise Exception("file does not exist")

    # creat dataframe for images and labels
    def make_df(self):
        images_df = pd.DataFrame(data=self.image_list, columns=['image_name', 'image_data'])
        images_df = images_df.set_index('image_name')
        labels_df = pd.read_csv(self.data_path + 'labels.txt', delim_whitespace=True, names=['image_name', 'x', 'y'])
        labels_df = labels_df.set_index('image_name')
        data_df = pd.concat([images_df, labels_df], axis=1, join_axes=[images_df.index])
        return data_df

    # prepare the train and test data
    # :return: train_test_split data and the input shape of image
    def modify_data(self, data_df):
        labels = []
        for i in range(len(data_df['x'].values)):
            # label value * 255 to make input and output have same area (0 - 255)
            label_x = data_df['x'].values[i] * 255
            label_y = data_df['y'].values[i] * 255
            labels.append([label_x, label_y])
        x_train, x_test, y_train, y_test = train_test_split(data_df['image_data'].values, labels,
                                                            test_size=0.07, random_state=20)
        x_train, x_test = np.stack(x_train, axis=0), np.stack(x_test, axis=0)  # x_train.shape = (num, 326, 490, 3)
        y_train, y_test = np.stack(y_train, axis=0), np.stack(y_test, axis=0)

        input_shape = self.get_input_shape()
        x_train = self.modify_image_shape(x_train)
        x_test = self.modify_image_shape(x_test)
        splited_data = [x_train, x_test, y_train, y_test]  # make them a list to simplify the param to return
        return splited_data, input_shape

    # change input_shape of data for channels_format may differ on different backends
    # backend using here is tensorflow, channels last
    def modify_image_shape(self, x):
        # if K.image_data_format() == 'channels_last':
        x = x.reshape(x.shape[0], self.im_rows, self.im_cols, self.channels)
        return x

    # return: the shape of cnn input layer
    def get_input_shape(self):
        # if K.image_data_format() == 'channels_last':
        input_shape = (self.im_rows, self.im_cols, self.channels)
        return input_shape
