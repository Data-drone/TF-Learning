import os
import glob
import cv2
import numpy as np

# process and reduce imagesize
def get_im(path, img_rows, img_cols):
    img = cv2.imread(path)
    resize = cv2.resize(img, (img_cols, img_rows) )
    
    return resized

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
            img = get_im(img_fl, 224, 224)
            X_train.append(img)
            y_train.append(j)
            #driver_id.append()

    return X_train, y_train


train_data, train_target = load_train()
train_data = np.array(train_data, dtype=np.uint8)
train_target = np.array(train_target, dtype=np.uint8)

train_data = train_data.reshape(train_data.shape[0], 1, 224, 224)
train_target = np_utils.to_categorical(train_target, 10)

train_data = train_data.astype('float32')
