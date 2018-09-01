from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D  # conv layer & pooling layer
from keras.layers import Flatten, Dense, Dropout, Activation  # fully connected layer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K


class Cnn(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.batch_size = 32
        self.epochs = 150
        self.learning_rate = 1e-2

    #  build CNN model by keras for training
    #  param: the shape of the first layer
    #  return: the keras model for training
    def build_net(self):
        model = Sequential()
        # layer 1: cov layer, 3*3*8
        model.add(Conv2D(filters=8, kernel_size=3, activation='relu', padding='valid', strides=2, input_shape=self.input_shape))
        model.add(MaxPool2D(pool_size=4))

        # layer 2: cov layer, 4*4*16
        model.add(Conv2D(filters=16, kernel_size=4, activation='relu', padding='valid'))
        model.add(MaxPool2D(pool_size=3))

        # layer 3: conv layer,4*4*32
        model.add(Conv2D(filters=32, kernel_size=4, padding='valid'))
        model.add(MaxPool2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # layer 4: dense layer
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # layer 5: dense layer
        model.add(Dense(units=256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        # layer 6: dense layer
        model.add(Dense(units=128, activation='relu'))
        model.add(BatchNormalization())

        # layer 7: dense layer
        model.add(Dense(units=32, activation='relu'))
        model.add(BatchNormalization())

        # layer 8: dense layer
        model.add(Dense(units=2, activation='linear'))
        model.summary()
        return model

    # compile and fit the model, show the train progress
    def train_test(self, model, splited_data):
        x_train, x_test = splited_data[0], splited_data[1]
        y_train, y_test = splited_data[2], splited_data[3]
        model.compile(loss=[self.distance_error], optimizer=Adam(lr=self.learning_rate), metrics=[self.distance_error])
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2, validation_data=(x_test, y_test))
        # model.save('find_phone_cnn.h5')

    #  define the loss function and evaluation metrics(评价函数， 结果不用于训练过程)
    @staticmethod
    def distance_error(y_true, y_pred):
        error = K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=1)), axis=0)
        return error