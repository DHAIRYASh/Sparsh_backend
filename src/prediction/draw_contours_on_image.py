import math
import os

import cv2
import numpy as np

from src.utils.utils import fetch, save


def draw_pred(path, exp_list, PREDS, prob):
    '''
    Draw bounding boxes and predictions on the image
    '''
    img = fetch(path)
    splits = path.split('.')
    font_size = (img.shape[0]*img.shape[1])/(576*1280)
    color = (255, 0, 0)
    path = os.path.join(path.replace(os.path.basename(path), ""), "image." + splits[-1])
    i = 0
    while i < len(exp_list):
        rect = cv2.minAreaRect(exp_list[i])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(img, [box], 0, color, 2)
        minx = math.inf
        miny = math.inf
        a = 0
        while a < len(box):
            if minx > box[a][0]:
                minx = box[a][0]
            if miny > box[a][1]:
                miny = box[a][1]
            a = a + 1
        cv2.putText(img, f'{PREDS[i]}: {prob[i] * 100:.2f}%', (minx + 5, miny + 60), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    color, 2)
        i += 1

    save(path, img)
    return path
