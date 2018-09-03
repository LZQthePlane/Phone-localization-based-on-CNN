'''
A trial with auto_keras API for auto_ML
While it raise Error "x_train and y_train should have the same number of instances" which confuse me
The shape of x_train and y_train is (479, 326, 490, 3) (479, 2), the instance is 479 the same
I check in AUTO-KERAS github and get a clue that "Y should be a one-dimensional array which has the same length as x"
but this information is for classification project while it's a regression project

I will try to work this out when auto_keras update it's official document
'''

import os
import numpy as np
from keras_sln import preprocess as p
from autokeras.image_supervised import ImageRegressor
from keras.models import load_model
from keras.utils import plot_model

model_save_path = os.path.dirname(os.path.abspath(__file__)) + os.sep


# train and test with autokeras API
# which can automatically find a best fitting model and hyperparameters during time_limit
def auto_train_test(splited_data):
    x_train, x_test = splited_data[0], splited_data[1]
    y_train, y_test = splited_data[2], splited_data[3]
    print(x_train.shape, y_train.shape)
    reg = ImageRegressor(verbose=True, augment=False)
    reg.fit(x_train, y_train, time_limit=10 * 60)  # the time for model searching is 10min
    reg.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    train_acc = reg.evaluate(x_train, y_train)
    test_acc = reg.evaluate(x_test, y_test)
    print('The train acc is {0:.4f}, the test acc is {1:.4f}'.format(train_acc, test_acc))
    return reg


# load the model out and plot it
def load_out_model(reg):
    reg.load_searcher().load_best_model().produce_keras_model().save(model_save_path)
    plot_model(load_model(model_save_path), to_file='auto_model.png')


if __name__ == '__main__':
    # preprocess the data for train and test
    process = p.DataProcess()
    process.import_images()
    data_df = process.make_df()
    splited_data, input_shape = process.modify_data(data_df)

    # auto search the model fitting best
    reg = auto_train_test(splited_data)
    load_out_model(reg)

