import os
from scipy.io import loadmat

import numpy as np
#import matplotlib.pyplot as plt

import cv2

cc = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

def rect_area(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])

def extract_lips(image):
    rects = cc.detectMultiScale(image, minNeighbors=1)
    return max(rects, key=rect_area)

LIPS_PATH = "/Users/eric/Downloads/avletters/Lips"

mouth_vids = []



for fname in os.listdir(LIPS_PATH):
    if fname.endswith(".mat"):
        vid = loadmat(LIPS_PATH + "/" + fname)['vid'].reshape((80, 60, -1)).transpose(1, 0, 2)
        print vid.shape

        frame = 0

        mouth = cv2.cvtColor(vid[:, :, frame], cv2.COLOR_GRAY2BGR)

        rects = cc.detectMultiScale(mouth, minNeighbors=1)
        """
        for rect in cc.detectMultiScale(mouth, minNeighbors=1):
            print rect
            cv2.rectangle(mouth, (rect[0], rect[1]), (rect[2], rect[3]), (125, 125, 25), 2)
        """

        mouth_rect = max(rects, key=rect_area)
        cv2.rectangle(mouth, (mouth_rect[0], mouth_rect[1]), (mouth_rect[2], mouth_rect[3]), (125, 125, 25), 1)
        print type(mouth_rect)
        print rect_area(mouth_rect)

        #plt.imshow(vid[:, :, 0], cmap=plt.cm.gray, interpolation=None)
        #plt.show()

        cv2.imshow('detected mouths', mouth)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



