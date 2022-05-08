import os
from pathlib import Path

from src.pre_process.data_augmentation import driver_aug
from src.pre_process.slice import drive_slice
from src.utils.utils import save, fetch


def get_path(path, boolean):
    '''
    Gets path of image and returns path to folder to save processed data
    '''
    fname = os.path.basename(path)
    imagen = fname.split(".")
    new_path = path.replace(fname, "")
    check = Path(path).parent.absolute()
    i = 0
    if boolean:
        while f"{imagen[0]}_{i}" in next(os.walk(check))[1]:
            i = i + 1
        imagen[0] = f"{imagen[0]}_{i}"
    return new_path, imagen


def data_augmentation(image, path):
    '''
    Augments the data
    '''
    aug_list = driver_aug(image)
    aug_type = ["rr", "hf", "vf", "gauss", "sp", "poison", "speckle"]
    new_path, imagen = get_path(path, False)
    i = 0
    for new_image in aug_list:
        save_path = os.path.join(new_path, f"{imagen[0]}_{aug_type[i]}.png")
        save(save_path, new_image)
        i = i + 1
    return None


def slicing(image, path):
    '''
    Creates slices of 32 by 32 for each image and saves them
    '''
    size = 32
    slice_list = drive_slice(image, size)
    new_path, imagen = get_path(path, True)
    path_ = os.path.join(new_path, imagen[0])
    os.mkdir(path_)
    save_dir = os.path.join(path_, "slice")
    os.mkdir(save_dir)
    i = 0
    for new_image in slice_list:
        save_path = os.path.join(save_dir, f"s{i}.png")
        save(save_path, new_image)
        i = i + 1
    return path_, imagen[1]


def driver_preprocess_train(parent_dir, boolean):
    '''
    Drives the preprocessing done in training pipeline
    '''
    path, dirs, files = next(os.walk(parent_dir))
    patha = list()
    for each in dirs:
        folder_path = os.path.join(parent_dir, each)
        path, dirss, files = next(os.walk(folder_path))
        if len(files) < 13:
            continue
        if boolean:
            images_path = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                           os.path.isfile(os.path.join(folder_path, f))]
            for path in images_path:
                if ".DS_Store" in path:
                    continue
                image = fetch(path)

                data_augmentation(image, path)

        images_path = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                       os.path.isfile(os.path.join(folder_path, f))]
        for path in images_path:
            if ".DS_Store" in path:
                continue
            image = fetch(path)
            if image.shape[0] > 512 or image.shape[1] > 512:
                image = fetch(path, (512, 512))
            path_, exten = slicing(image, path)
            patha.append(path_)
            save_path = os.path.join(path_, "image.png")
            save(save_path, image)
            os.remove(path)
    return patha


def driver_preprocess_pred(parent_dir):
    '''
    Drives the preprocessing required for prediction pipeline
    '''
    patha = list()
    files = next(os.walk(parent_dir))[2]
    for each in files:
        path = os.path.join(parent_dir, each)
        if ".DS_Store" in path or ".npy" in path:
            continue
        image = fetch(path)
        if image.shape[0] > 512 or image.shape[1] > 512:
            image = fetch(path, (512, 512))
        path_, exten = slicing(image, path)
        patha.append(path_)
        save_path = os.path.join(path_, "image.png")
        save(save_path, image)
        os.remove(path)
    return patha
