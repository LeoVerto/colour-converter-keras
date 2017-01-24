#!/usr/bin/env python3

import numpy as np
import utils


def generate_sample(v=False):
    rgb = np.random.rand(1, 3)[0]
    hsv = utils.pixel_rgb2hsv(rgb)
    lab = utils.pixel_rgb2lab(rgb)

    if v:
        print(utils.rgb2string(rgb))
        print(utils.hsv2string(hsv))
        print(utils.lab2string(lab))

    return np.concatenate([rgb, hsv, lab])


def generate_samples(n, v=False):
    samples = []
    for i in range(n):
        samples.append(generate_sample(v))

    return np.array(samples)


def save_samples(n, file):
    samples = generate_samples(n)
    np.save(file, samples)


def test():
    s = generate_samples(1, True)
    s = s[0]
    print(s)

save_samples(55000, "training.npy")
save_samples(5000, "validation.npy")
