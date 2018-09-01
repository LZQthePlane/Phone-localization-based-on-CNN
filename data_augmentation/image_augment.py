import os
import glob
import cv2 as cv
import pandas as pd


file_path = os.path.dirname(os.path.abspath(__file__))  # file path of this project


# import images and labels
def import_images():
    if os.path.exists(file_path):
        print("file_path exist")
        files = glob.glob(file_path + os.sep + 'find_phone' + os.sep + '*.jpg')
        return files
    else:
        raise Exception('image_file_path does not exist')


# flip images horizontally, vertically, diagonally
def flip_images(images):
    save_path = file_path + os.sep + 'flip_images' + os.sep + 'd_flip'
    for image in images:
        im = cv.imread(image)
        im_name = image.split(os.sep)[-1]
        im_h = cv.flip(im, flipCode=1, dst=None)  # Flip horizontally
        im_v = cv.flip(im, flipCode=0, dst=None)  # Flip vertically
        im_d = cv.flip(im, flipCode=-1, dst=None)  # Flip diagonally
        # cv.imshow('h_flip', im_cp)
        # cv.waitKey()
        im_origin_num = im_name.split('.')[0]
        h_flip_name = im_origin_num + '_h' + '.jpg'
        v_flip_name = im_origin_num + '_v' + '.jpg'
        d_flip_name = im_origin_num + '_d' + '.jpg'

        cv.imwrite(os.path.join(save_path, h_flip_name), im_h)
        cv.imwrite(os.path.join(save_path, v_flip_name), im_v)
        cv.imwrite(os.path.join(save_path, d_flip_name), im_d)
        cv.destroyAllWindows()


# change the position(x, y) along with the flip
def flip_labels():
    old_file = file_path + os.sep + 'find_phone' + os.sep + 'labels.txt'
    labels_df = pd.read_csv(old_file, delim_whitespace=True, names=['image_name', 'x', 'y'])
    im_num = len(labels_df['image_name'])
    for i in range(im_num):
        # change the name of images
        old_name = labels_df['image_name'].values[i]
        new_name = old_name.split('.')[0] + '_d' + '.jpg'
        labels_df['image_name'].values[i] = new_name
        # change the x_position due to flip
        labels_df['x'].values[i] = str(1 - float(labels_df['x'].values[i]))
        labels_df['y'].values[i] = str(1 - float(labels_df['y'].values[i]))
    print(labels_df)
    labels_df.to_csv('labels_d.txt', sep=' ', index=False, float_format='%.4f')


if __name__ == "__main__":
    images = import_images()
    # flip_images(images)
    # flip_labels()
