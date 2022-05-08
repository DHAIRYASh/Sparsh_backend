import random

import numpy as np
# noinspection PyUnresolvedReferences
from scipy import ndarray
from skimage.transform import rotate


def random_rotation(image: ndarray):
    '''
    pick a random degree of rotation between 25% on the left and 25% on the right
    '''
    random_degree = random.uniform(-25, 25)
    return rotate(image, random_degree)


def horizontal_flip(image: ndarray):
    '''
    Horizontally flip the image
    '''
    return image[:, ::-1]


def vertical_flip(image):
    '''
    Vertically flip the image
    '''
    return np.rot90(image, 2)


def noise(image):
    '''
    Adds various noises to the image
    '''
    def gauss(image):
        '''
        Adds Gaussian noise to the image
        '''
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    def sp(image):
        '''
        Adds s&p noise to the image
        '''
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    def poisson(image):
        '''
        adds poisson noise to the image
        '''
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    def speckle(image):
        '''
        adds speckle noise to the image
        '''
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy

    return gauss(image), sp(image), poisson(image), speckle(image)


def driver_aug(image):
    '''
    Calls nosie function for data augmentation
    '''
    rr = random_rotation(image)
    hf = horizontal_flip(image)
    vf = vertical_flip(image)
    gauss, sp, poison, speckle = noise(image)
    aug_list = [rr, hf, vf, gauss, sp, poison, speckle]
    return aug_list
