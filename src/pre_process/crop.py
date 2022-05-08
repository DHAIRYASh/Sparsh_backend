import os
import re

import cv2
import numpy as np
import pandas as pd

from src.utils.utils import save, fetch


def save_image(name, dtype, image, folder, i=0, boolean=True):
    '''
    Save the cropped image in the folder
    '''
    parent_folder = os.path.dirname(folder)
    folder = os.path.basename(folder)
    if boolean:
        if folder not in os.listdir(parent_folder):
            os.mkdir(os.path.join(parent_folder, folder))
    if dtype is not None:
        d_folder = os.path.join(parent_folder, folder, dtype)
        if dtype not in os.listdir(os.path.join(parent_folder, folder)):
            os.mkdir(d_folder)
    else:
        d_folder = os.path.join(parent_folder, folder)
    name = name.split('.')
    name = f"{name[0]}_{i}.png"
    save_path = os.path.join(d_folder, name)
    i = 0
    while True:
        path, exten = save_path.split(".")
        save_path = f"{path}_{i}.{exten}"
        if not os.path.exists(save_path):
            break
        i = i + 1
    save(save_path, image)
    return None


def get_cords_train(df):
    '''
    Get coordinates and labels of the bounding box
    '''
    exp_list = list()
    typ_list = list()
    for ind in range(len(df)):
        typ = df["body"][ind][0]['value']
        test = re.sub('[^0-9,. ]+', '', df['target'][ind]['selector']['value'])
        tst = test.split(' ')
        lst = [tst[1:][i].split(',') for i in range(len(tst) - 1)]
        clist = [[round(float(lst[i][0])), round(float(lst[i][1]))] for i in range(len(lst))]
        exp_list.append(np.asarray(clist))
        typ_list.append(typ)
    return typ_list, exp_list


def get_cords_predict(df):
    '''
    Get coordinates of the bounding box
    '''
    final_lst = list()
    for ind in range(len(df)):
        test = re.sub('[^0-9,. ]+', '', df['target'][ind]['selector']['value'])
        tst = test.split(' ')
        lst = [tst[1:][i].split(',') for i in range(len(tst) - 1)]
        exp_list = [[round(float(lst[i][0])), round(float(lst[i][1]))] for i in range(len(lst))]
        final_lst.append(np.asarray(exp_list))
    return final_lst


def crop(img, rect):
    '''
    Crops the image based on the bounding box
    '''
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def crop_and_save(annotation_folder, name, typ, exp_list, folder_out):
    '''
    Calls crop and save and gives them parameters to work
    '''
    if typ is not None:
        image_path = name
    else:
        image_path = os.path.join(annotation_folder, name)
    image = fetch(image_path)
    for i in range(len(exp_list)):
        crop_details = cv2.minAreaRect(exp_list[i])
        if typ is not None:
            dtype = typ[i].lower()
        else:
            dtype = None
        cropped = crop(image, crop_details)
        if typ is not None:
            save_image(os.path.basename(name), dtype, cropped, folder_out, i)
        else:
            save_image(name, dtype, cropped, folder_out, i, False)
    return None


def driver_crop_pred(annotation_folder, folder_out):
    '''
    Reads the json file and calls crop and save for prediction pipeline
    '''
    for i in os.listdir(annotation_folder):
        if '.json' in i:
            file = i
        else:
            name_2 = i
    folder_path = os.path.join(annotation_folder, file)
    json_path = os.path.join(folder_path)
    df = pd.read_json(json_path)
    print("df read")
    name, exp_list = name_2, get_cords_predict(df)
    print("got cords")
    typ = None
    crop_and_save(annotation_folder, name, typ, exp_list, folder_out)
    print("cropped and saved")
    return exp_list


def driver_crop_train(image_path_list, json_path_list, folder_out):
    '''
    Reads the json file and calls crop and save for training pipeline
    '''
    imgfn = image_path_list
    jsonfn = json_path_list
    for i in range(len(jsonfn)):
        df = pd.read_json(jsonfn[i])
        name = imgfn[i]
        typ, exp_list = get_cords_train(df)
        crop_and_save("", name, typ, exp_list, folder_out)
    return None
