"""
PFE Boulet Olgiati
Created :      12/03/23
Last update :  12/03/23
"""

import os
import json
import struct
import random
import numpy as np


def get_datasets(data_path) : 

    with open(os.path.join(data_path, 'train-images.idx3-ubyte'),'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_images = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_images = train_images.reshape((size, nrows, ncols))
    f.close()

    with open(os.path.join(data_path, 'train-labels.idx1-ubyte'),'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size,))
    f.close()

    with open(os.path.join(data_path, 't10k-images.idx3-ubyte'),'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_images = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_images = test_images.reshape((size, nrows, ncols))
    f.close()

    with open(os.path.join(data_path, 't10k-labels.idx1-ubyte'),'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size,))
    f.close()

    return train_images, train_labels, test_images, test_labels


def normalize_images(images) : 
    return (images - np.mean(images)) / np.std(images)


def images_to_TS(images, labels) : 

  n_images, img_size, _ = np.shape(images)
  T = n_images * img_size
  U = np.zeros((T, img_size))
  V = np.zeros((T, 10))

  for i in range(n_images) : 
    U[i*img_size : (i+1)*img_size] = images[i]
    for j in range(img_size) : 
      V[i*img_size + j][labels[i]] = 1

  return U, V


def main(data_path) : 

    train_images, train_labels, test_images, test_labels = get_datasets(data_path)

    # Normalize training and test images
    normalized_train_images = normalize_images(train_images)
    normalized_test_images = normalize_images(test_images)

    # Convert datasets to time series
    U_train, V_train = images_to_TS(normalized_train_images, train_labels)
    U_test, V_test = images_to_TS(normalized_test_images, test_labels)

    return U_train, V_train, U_test, V_test


if __name__=='__main__' : 

    with open("config.json", "r") as f:
        config = json.load(f)

    SEED = config["SEED"]
    config_data = config["data"]
    data_path = config_data["DATA_PATH"]

    main(data_path)