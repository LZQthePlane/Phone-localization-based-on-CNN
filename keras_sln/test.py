import os
import cv2 as cv
from keras.models import load_model
from keras_sln import keras_net
import keras_sln.preprocess as p


im_rows, im_cols = 326, 490
image_path = os.path.dirname(os.path.abspath(__file__))+os.sep+'test_image.jpg'


# load the pre_trained model
def load_trainedmodel():
    distance_error = keras_net.Cnn.distance_error
    model = load_model('find_phone_cnn.h5', custom_objects={'distance_error': distance_error})
    return model


# load the image for test
def load_test_image():
    if os.path.exists(image_path):
        print("test image exists")
        im = cv.imread(image_path)
        im = cv.resize(im, (im_cols, im_rows), interpolation=cv.INTER_CUBIC)
        return im
    else:
        raise Exception("test image does not exist")


# reshape the test image for channel difference
def reshape_image(im):
    process = p.DataProcess()
    im_for_pred = im.reshape(1, -1)  # make first dimension the num of images
    im_modified = process.modify_image_shape(im_for_pred)
    return im_modified


# calculate the predict (x, y) position
def get_prediction(model, im):
    result = model.predict(im)
    # result / 255 to restore the true value of position
    x = round(result[0][0]/255, 4)  # 四舍五入， 保留4位小数
    y = round(result[0][1]/255, 4)
    print('The position of phone in test image is ({0}, {1})'.format(x, y))
    return x, y


# plot the position on image, feel the difference of true and prediction
def plot(im, x, y):
    #  *490 and *326 to to plot the position in image
    x_im = int(round(x * im_cols, 0))
    y_im = int(round(y * im_rows, 0))
    cv.line(im, (0, y_im), (im_cols, y_im), color=(127, 255, 0), thickness=1)
    cv.line(im, (x_im, 0), (x_im, im_rows), color=(127, 255, 0), thickness=1)
    cv.imshow('test_image', im)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    model = load_trainedmodel()
    im = load_test_image()
    im_modified = reshape_image(im)
    x, y = get_prediction(model, im_modified)
    plot(im, x, y)