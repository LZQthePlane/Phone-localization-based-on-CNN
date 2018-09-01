import os
import cv2 as cv
from tensorflow_sln.tf_model import tf_net
import tensorflow as tf

im_cols, im_rows, channels = 490, 326, 3
image_path = os.path.dirname(os.path.abspath(__file__))+os.sep+'test_image.jpg'
model_save_path = os.path.dirname(os.path.abspath(__file__))+os.sep+'model_save'+os.sep


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
    im_for_pred = im.reshape(1, -1)  # make first dimension the num of images
    return im_for_pred


# plot the position on image, feel the difference of true and prediction
def plot(im, x, y):
    x_im, y_im  = int(round(x * im_cols, 0)), int(round(y * im_rows, 0))
    cv.line(im, (0, y_im), (im_cols, y_im), color=(127, 255, 0), thickness=1)
    cv.line(im, (x_im, 0), (x_im, im_rows), color=(127, 255, 0), thickness=1)
    cv.imshow('test_image', im)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # load the meta graph and model
    sess = tf_net.start_sess()
    saver = tf.train.import_meta_graph(model_save_path + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(model_save_path))

    # load the test image
    im = load_test_image()
    im_modified = reshape_image(im)

    # restore the tensor for prediction
    graph = tf.get_default_graph()
    x_hold = graph.get_tensor_by_name("x_hold:0")
    with_bn = graph.get_tensor_by_name('with_bn:0')
    pred = tf.get_collection('predict')

    # compute the predict (x, y) position
    result = sess.run(pred, feed_dict={x_hold: im_modified, with_bn:False})
    x = round(result[0][0][0] / 255, 4)
    y = round(result[0][0][1] / 255, 4)
    print('The position of the phone in test image is ({0}, {1})'.format(x, y))
    plot(im, x, y)