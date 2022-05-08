import os
import shutil

from src.utils.utils import data_handle_dict, fetch, data_and_models


def gen_val_data(mid_dir, val_dir):
    '''
    Generates validation data from the mid_dir
    '''
    if os.path.basename(val_dir) not in os.listdir(data_and_models):
        os.mkdir(val_dir)
    if os.path.basename(mid_dir) not in os.listdir(data_and_models):
        os.mkdir(mid_dir)
    if not os.path.exists(data_handle_dict):
        boolean = True
        dic = {}
    else:
        boolean = False
        dic = fetch(data_handle_dict)

    for e in os.listdir(mid_dir):
        to_folder = os.path.join(val_dir, e)
        from_folder = os.path.join(mid_dir, e)
        if not os.path.exists(to_folder):
            os.mkdir(to_folder)
        from_files = [os.path.join(from_folder, f) for f in os.listdir(from_folder)]
        to_files = [os.path.join(to_folder, f) for f in os.listdir(from_folder)]
        files_in_to_folder = [os.path.join(to_folder, f) for f in os.listdir(to_folder)]
        if boolean:
            dic[from_folder] = False
        if dic[from_folder]:
            if len(files_in_to_folder) == 12:
                for f in files_in_to_folder:
                    os.remove(f)
            i = 0
            while i < len(from_files):
                j = 0
                while True:
                    name, exten = to_files[i].split(".")
                    to_files[i] = f"{name}_{j}.{exten}"
                    if to_files[i] not in files_in_to_folder:
                        break
                    j += 1
                shutil.copyfile(from_files[i], to_files[i])
                i += 1
        else:
            for f in os.listdir(to_folder):
                f = os.path.join(to_folder, f)
                os.remove(f)
            i = 0
            while i < len(from_files):
                j = 0
                while to_files[i] in files_in_to_folder:
                    name, exten = to_files[i].split(".")
                    to_files[i] = f"{name}_{j}.{exten}"
                    j += 1
                shutil.copyfile(from_files[i], to_files[i])
                i += 1
    return None
