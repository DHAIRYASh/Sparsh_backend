import os
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from src.utils.utils import mid_dir_path as mid_data, data_handle_dict, data_and_models
from src.utils.utils import save, fetch

path_d = os.path.join(data_and_models, "dict")
dic_e_g = None
dic_d_g = None
dic_le_g = None
dic_ld_g = None
dict_e_p = os.path.join(path_d, "dict_e.pkl")
dict_d_p = os.path.join(path_d, "dict_d.pkl")
dict_le_p = os.path.join(path_d, "dict_le.pkl")
dict_ld_p = os.path.join(path_d, "dict_ld.pkl")


def encode_in(y, dic_e):
    '''
    Onehot-encodes the input y.
    '''
    y = [dic_e[e] for e in y]
    return y


def decode_in(y, dic_d):
    '''
    Decodes the input y.
    '''
    y = [dic_d[str(e)] for e in y]
    return y


def make_dict():
    '''
    Make dictionary for encoding and decoding.
    '''
    try:
        dic = fetch(data_handle_dict)
        e = list(dic.keys())
        e = [os.path.basename(f) for f in e]
    except:
        e = os.listdir(mid_data)

    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    e_le = label_encoder.fit_transform(e)
    e_e = e_le.reshape(len(e_le), 1)
    e_e = onehot_encoder.fit_transform(e_e)
    dic_e = {}
    dic_d = {}
    dic_le = {}
    dic_ld = {}
    i = 0
    while i < len(e):
        dic_e[e[i]] = e_e[i]
        dic_d[str(e_e[i])] = e[i]
        dic_le[e[i]] = e_le[i]
        dic_ld[str(e_le[i])] = e[i]
        i = i + 1
    return dic_e, dic_d, dic_le, dic_ld


def compare_dict():
    '''
    Compare and update the one-hot-encoding dictionary if needed.
    '''
    dic_e, dic_d, dic_le, dic_ld = make_dict()
    dirs = os.listdir(data_and_models)
    if os.path.basename(path_d) not in dirs:
        os.mkdir(path_d)
        save(dict_e_p, dic_e)
        save(dict_d_p, dic_d)
        save(dict_le_p, dic_le)
        save(dict_ld_p, dic_ld)
        return dic_e
    dic_d_o = fetch(dict_d_p)
    if dic_d != dic_d_o:
        n_p = os.path.join(data_and_models, "numpy_data")
        if n_p not in dirs:
            save(dict_e_p, dic_e)
            save(dict_d_p, dic_d)
            save(dict_le_p, dic_le)
            save(dict_ld_p, dic_ld)
            return dic_e
        for each in os.listdir(n_p):
            if "y_" in each:
                path = os.path.join(n_p, each)
                y = fetch(path)
                y = decode_in(y, dic_d_o)
                y = encode_in(y, dic_e)
                save(path, y)
        save(dict_e_p, dic_e)
        save(dict_d_p, dic_d)
        save(dict_le_p, dic_le)
        save(dict_ld_p, dic_ld)
    return dic_e


def get_class(path):
    '''
    Get the class of the given path.
    '''
    p1 = Path(path).parent.absolute()
    p2 = Path(p1).parent.absolute()
    p1 = str(p1)
    p2 = str(p2)
    c = p1.replace(f"{p2}\\", "").lower()
    return c


def encode(y, boolean):
    '''
    Gets class and encodes y.
    '''
    dic_e = compare_dict()
    if boolean:
        y = [get_class(e) for e in y]
    y = encode_in(y, dic_e)
    return y


def decode(y):
    '''
    Decodes One-hot encoded y.
    '''
    dic_d = fetch(dict_d_p)
    y = decode_in(y, dic_d)
    return y


def lencode(y):
    '''
    Label-encodes y.
    '''
    dic_le = fetch(dict_le_p)
    y = encode_in(y, dic_le)
    return y


def ldecode(y):
    '''
    Decodes label-encoded y.
    '''
    dic_ld = fetch(dict_ld_p)
    y = decode_in(y, dic_ld)
    return y


def get_num_classes():
    '''
    Get number of classes.
    '''
    dic = fetch(dict_e_p)
    return len(dic)
