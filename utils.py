#!/usr/bin/env python3

import numpy as np
from skimage import color


def pixel_rgb2hsv(rgb):
    return color.rgb2hsv(np.reshape(rgb, (1, 1, 3)))[0][0]


def pixel_rgb2lab(rgb):
    return color.rgb2lab(np.reshape(rgb, (1, 1, 3)))[0][0]


def rgb2string(rgb):
    return "RGB: {}".format(rgb*255)


def hsv2string(hsv):
    return "HSV: {}, {}, {}".format(hsv[0] * 360, hsv[1] * 100, hsv[2] * 100)


def lab2string(lab):
    return "Lab: {}, {}, {}".format(lab[0], lab[1], lab[2])


def upscale_lab(lab):
    mul = np.array([50, 128, 128])
    add = np.array([50, 0, 0])

    return lab * mul + add
