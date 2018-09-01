import os
import tensorflow_sln.preprocess as p
from tensorflow_sln.tf_model import tf_net


train_model_save_path = os.path.dirname(os.path.abspath(__file__))+os.sep+'model_save/'

if __name__ == "__main__":
    # preprocess the data for train and test
    process = p.DataProcess()
    process.import_images()
    data_df = process.make_df()
    splited_data, input_shape = process.modify_data(data_df)

    # creat cnn model, train and test accuracy
    cnn = tf_net.Cnn(input_shape)
    pred, x_hold, y_hold, keep_prob, with_batchnorm = cnn.build_net()
    cnn.train_test(pred, x_hold, y_hold, keep_prob, with_batchnorm, splited_data, train_model_save_path)

    # cnn = tf_net_eager.Cnn(input_shape)
    # pred, x_hold, y_hold, keep_prob, with_batchnorm = cnn.build_net()
    # cnn.train_test(pred, x_hold, y_hold, keep_prob, with_batchnorm, splited_data, train_model_save_path)
