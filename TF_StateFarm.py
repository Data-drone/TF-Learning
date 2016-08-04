import os
import glob
import cv2
import numpy as np
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# process and reduce imagesize
def get_im(path, img_rows, img_cols):
    img = cv2.imread(path)
    resize = cv2.resize(img, (img_cols, img_rows) )
    
    return resize

# Load the training data
def load_train():
    X_train = []
    y_train = []
    driver_id = []
    # need to get driver data?
    
    print('read train images')
    for j in range(10):
        folder = 'c' + str(j)
        
        path = os.path.join('StateFarm', 'train', folder, '*.jpg')
        files = glob.glob(path)
        for img_fl in files:
            flbase = os.path.basename(img_fl)
            img = get_im(img_fl, IMAGE_SIZE, IMAGE_SIZE)
            X_train.append(img)
            y_train.append(j)
            #driver_id.append()

    return X_train, y_train


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

train_data, train_target = load_train()
train_data = np.array(train_data, dtype=np.uint8)
train_target = np.array(train_target, dtype=np.uint8)

#train_data = train_data.reshape(train_data.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE)
train_data = train_data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
train_target = to_categorical(train_target, NUM_CLASSES)

train_data = train_data.astype('float32')





