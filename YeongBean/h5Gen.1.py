import numpy as np
import h5py

import glob
import cv2

#shape = [data, label]

hdf5_path = 'data/VOC2012.h5'  # address to where you want to save the hdf5 file
train_path = 'data\VOC2012\*.jpg' #address to where you want to create hdf5 file

# read addresses and labels from the 'train' folder
addrs = glob.glob(train_path)
labels = [file for file in addrs if file.endswith(".jpg")]
    
# Divide the hata into 60% train, 20% validation, and 20% test
train_labels = labels[0:int(0.1* len(labels))]

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow


# check the order of data and chose proper data shape to save images
if data_order == 'th':
    data_shape = (0, 3, 256, 256)
elif data_order == 'tf':
    data_shape = (0, 256, 256, 3)

# open a hdf5 file and create earrays

train_storage = []
imglabels = []
# create the label arrays and copy the labels data in them
#hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)

# a numpy array to save the mean of the images
index = 0
# loop over train addresses
for i in range(len(addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print ("Train data: {}/{}".format(i, len(addrs)))

    # read an image and resize to (256, 256)
    # cv2 load images as BGR, convert it to RGB
    addr = addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add any image pre-processing here

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    # save the image and calculate the mean so far
    train_storage.append(img[None])
    imglabels.append(index + 1)

# save the mean and close the hdf5 file

with h5py.File(hdf5_path, 'w') as hdf:
    hdf.create_dataset('data', data = train_storage)
    hdf.create_dataset('label', data = imglabels)