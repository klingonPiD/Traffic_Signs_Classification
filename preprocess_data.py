"""Utility file to perform pre-processing on input images"""

import numpy as np
import cv2

def compute_train_mean(image_data_train):
    """Returns mean image"""
    mean_img = np.mean(image_data_train, axis=0)
    return mean_img

def compute_train_statistics(image_data_train):
    """Returns
        min: min val of training samples
        max: max val of training samples"""
    X_min, X_max = np.min(image_data_train), np.max(image_data_train)
    return (X_min, X_max)



def subtract_mean(image_data, mean_img):
    image_data[:,...] -= mean_img
    return image_data


def min_max_scaling(image_data, X_min, X_max):
    a, b = -1.0, 1.0
    image_data[:, ...] = a + ((image_data - X_min) * (b - a)) / (X_max - X_min)
    return image_data

def hist_eq(image_data):
    image_data_out = np.zeros_like(image_data)
    for i in range(len(image_data)):
        img = image_data[i,...]#np.reshape(image_data[i,...], (32, 32, 3))
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        y, cr, cb = cv2.split(img_ycrcb)
        y = cv2.equalizeHist(y)
        img_ycrcb = cv2.merge((y, cr, cb))
        img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)
        image_data_out[i,...] = img#np.reshape(img,(1,32*32*3))
    return image_data_out

