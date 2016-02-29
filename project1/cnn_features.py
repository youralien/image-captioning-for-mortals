"""
=====================================
compute cnn features for coco dataset
=====================================
"""

# from sklearn_theano.feature_extraction import OverfeatTransformer
# tf = OverfeatTransformer(output_layers=[-3])

import time
import os
import numpy as np
from PIL import Image

import config

def load_images_to_matrix(file_paths, size=(231,231)):
    image_count = len(file_paths)
    X = np.zeros((image_count, size[0], size[1], 3), dtype='uint8')
    count = 0
    for f in file_paths:
        try:
            # Some of these files are not reading right...
            im = Image.open(f, 'r')
            X[count] = np.array(im.resize(size))
            count += 1
        except:
            continue
    return X

def test_load_images_to_matrix():
    img_dir = "{}/images/train2014/".format(config.COCO_DIR)

    fns = os.listdir(img_dir)[:10]
    full_fns = [os.path.join(img_dir,fn) for fn in fns]

    X = load_images_to_matrix(full_fns)
    print X.shape

def cache_image_features(img_dir, dest_dir):

    if not os.path.exists(dest_dir):
        # FIX: mkdir not working
        os.mkdir(dest_dir)

    fns = os.listdir(img_dir)
    full_fns = [os.path.join(img_dir,fn) for fn in fns]

    for i, fn in enumerate(fns):
        features = analyze(img_dir + fn)
        fn_w_out_extension = fn.split('.')[0]
        np.save(dest_dir+fn_w_out_extension, features)

def cache_train():
    cache_image_features(
          "{}/images/train2014/".format(config.COCO_DIR)
        , "{}/features/train2014/".format(config.COCO_DIR))

def cache_valid():
    cache_image_features(
          "{}/images/val2014/".format(config.COCO_DIR)
        , "{}/features/val2014/".format(config.COCO_DIR))

if __name__ == '__main__':
    test_load_images_to_matrix()
