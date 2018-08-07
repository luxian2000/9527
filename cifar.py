#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
data_1 = batch_1[b'data']

k = 3
pic = np.array([])
for i in range(1024):
    pic = np.append(pic, [data_1[k,i+2048], data_1[k,i+1024], data_1[k,i]])
pic=pic.reshape(32,32,3)

cv2.imwrite('cifar_pic.jpg', pic)
