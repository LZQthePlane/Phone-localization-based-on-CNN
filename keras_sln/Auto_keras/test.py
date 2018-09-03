'''
A test script for auto_keras
It works here for the mnist demo, while raise bug in auto_model
'''


from keras.datasets import mnist
import numpy as np
from autokeras.image_supervised import ImageClassifier


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    print(x_train.shape, y_train.shape, type(y_train))
    y_train = np.reshape(y_train, (len(y_train), -1))
    # print(y_train.shape)
    # y_train = np.reshape(y_train, (len(y_train),))
    print(y_train.shape)

    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=12*60)
    clf.final_fit(x_train, y_train, x_test, y_test,retrain=True)
    y = clf.evaluate(x_test, y_test)
    print(y * 100)
