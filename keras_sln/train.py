import numpy as np
import keras_sln.preprocess as p
from keras_sln import keras_net


# return: the accuracy of giving-test data
def cal_pred_acc(y_test, y_pred):
    dis = y_test - y_pred
    corrcet_num = 0
    for dis_each in dis:
        dis_each_x, dis_each_y = dis_each[0], dis_each[1]
        if np.sqrt((dis_each_x/255) ** 2 + (dis_each_y/255) ** 2) < 0.05:
            corrcet_num += 1
    accuracy = float(corrcet_num) / len(y_test)
    return accuracy


# calculate accuracy and print the results
def print_acc(cnn_model, splited_data):
    x_train, x_test = splited_data[0], splited_data[1]
    y_train, y_test = splited_data[2], splited_data[3]
    train_pred = cnn_model.predict(x_train)
    train_acc = cal_pred_acc(y_train, train_pred)
    print('Training completed')
    print('Train accuracy is {0}'.format(train_acc))

    test_pred = cnn_model.predict(x_test)
    test_acc = cal_pred_acc(y_test, test_pred)
    print('Test accuracy is {0}'.format(test_acc))


if __name__ == "__main__":
    # preprocess the data for train and test
    process = p.DataProcess()
    process.import_images()
    data_df = process.make_df()
    splited_data, input_shape = process.modify_data(data_df)

    # creat cnn model, train and test accuracy
    cnn = keras_net.Cnn(input_shape)
    cnn_model = cnn.build_net()
    cnn.train_test(cnn_model, splited_data)
    print_acc(cnn_model, splited_data)
