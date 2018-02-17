#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from keras import backend as K
from colormath.color_diff_matrix import delta_e_cie2000

PI_TENSOR = K.constant(np.pi)
ZERO = K.constant(0)
ONE = K.constant(1)
TWO = K.constant(2)
SEVEN = K.constant(7)
THREESIXTY = K.constant(360)
ONEEIGHTY = K.constant(180)


# Delta E 76 using NumPy
def cie1976_np(y_true, y_pred):
    distance = np.sqrt(
        np.square(y_pred[:, 0]-y_true[:, 0])
        + np.square(y_pred[:, 1]-y_true[:, 1])
        + np.square(y_pred[:, 2]-y_true[:, 2])
    )
    # print(distance)
    return distance


# Delta E 76 using Keras
def cie1976_keras(y_true, y_pred):
    distance = K.sqrt(
        K.square(y_pred[:, 0]-y_true[:, 0])
        + K.square(y_pred[:, 1]-y_true[:, 1])
        + K.square(y_pred[:, 2]-y_true[:, 2])
    )
    return distance


# Delta E 2000 using NumPy
def cie2000_np(y_true, y_pred):
    return delta_e_cie2000(y_true, y_pred)


def deg2rad(deg):
    return deg * (PI_TENSOR/ONEEIGHTY)


def rad2deg(rad):
    return rad * (ONEEIGHTY/PI_TENSOR)


# Delta E 2000 using Keras
# https://github.com/gtaylor/python-colormath/blob/master/colormath/color_diff_matrix.py#L112
# noinspection PyPep8Naming
def cie2000_keras(y_true, y_pred, Kl=1, Kc=1, Kh=1):
    L1 = y_true[0][0]
    a1 = y_true[0][1]
    b1 = y_true[0][2]

    L2 = y_pred[0][0]
    a2 = y_pred[0][1]
    b2 = y_pred[0][2]

    avg_Lp = (L1 + L2) / TWO

    C1 = K.sqrt(K.pow(a1 + b1, 2))
    C2 = K.sqrt(K.pow(a2 + b2, 2))

    avg_C1_C2 = (C1 + C2) / TWO

    G = K.constant(0.5) * (1 - K.sqrt(K.pow(avg_C1_C2, SEVEN) / (K.pow(avg_C1_C2, SEVEN) + K.pow(K.constant(25.0), SEVEN))))

    a1p = (ONE + G) * a1
    a2p = (ONE + G) * a2

    C1p = K.sqrt(K.pow(a1p, 2) + K.pow(b1, 2))
    C2p = K.sqrt(K.pow(a2p, 2) + K.pow(b2, 2))

    avg_Cp = (C1p + C2p) / TWO

    h1p = rad2deg(tf.atan2(b1, a1p))
    h1p = K.switch(K.less(h1p, ZERO), h1p + THREESIXTY, h1p)

    h2p = rad2deg(tf.atan2(b2, a2p))
    h2p = K.switch(K.less(h2p, ZERO), h2p + THREESIXTY, h2p)

    diff_h2p_h1p = h2p - h1p
    delta_hp = K.switch(K.greater(K.abs(h1p - h2p), ONEEIGHTY), diff_h2p_h1p + THREESIXTY, diff_h2p_h1p)
    delta_hp = K.switch(K.greater(h2p, h1p), delta_hp - K.constant(720), delta_hp)

    avg_Hp = K.switch(K.greater(K.abs(h1p - h2p), ONEEIGHTY), THREESIXTY + h1p + h2p, h1p + h2p) / TWO

    T = 1 - K.constant(0.17) * K.cos(deg2rad(avg_Hp - 30)) + \
        K.constant(0.24) * K.cos(deg2rad(2 * avg_Hp)) + \
        K.constant(0.32) * K.cos(deg2rad(3 * avg_Hp + 6)) - \
        K.constant(0.2) * K.cos(deg2rad(4 * avg_Hp - 63))

    S_L = 1 + ((K.constant(0.015) * K.pow(avg_Lp - 50, 2)) / K.sqrt(20 + K.pow(avg_Lp - 50, TWO)))
    S_C = 1 + K.constant(0.045) * avg_Cp
    S_H = 1 + K.constant(0.015) * avg_Cp * T

    delta_ro = 60 * K.exp(-(K.pow(((avg_Hp - 275) / 25), TWO)))
    R_C = K.sqrt((K.pow(avg_Cp, SEVEN)) / (K.pow(avg_Cp, SEVEN) + K.pow(K.constant(25.0), SEVEN)))
    R_T = -2 * R_C * K.sin(deg2rad(delta_ro))

    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p
    delta_Hp = 2 * K.sqrt(C2p * C1p) * K.sin(deg2rad(delta_hp) / TWO)

    return K.sqrt(
        K.pow(delta_Lp / (S_L * Kl), 2) +
        K.pow(delta_Cp / (S_C * Kc), 2) +
        K.pow(delta_Hp / (S_H * Kh), 2) +
        R_T * (delta_Cp / (S_C * Kc)) * (delta_Hp / (S_H * Kh)))


def test_delta_e():
    a = np.array([47, 26, -23])
    b = np.array([47, 26, -24])
    c = np.array([58, -28, 18])

    a = np.array([.1, .5, -.4])
    b = np.array([.9, .6, -.56])
    c = np.array([.3, -.5, .12])

    a3 = np.array([a, a, a])
    b3 = np.array([b, b, b])
    c3 = np.array([c, c, c])

    print("1976:")
    print("Numpy: {} Keras: {}".format(cie1976_np(a3, a3), K.eval(cie1976_keras(K.variable(a3), K.variable(a3)))))
    print("Numpy: {} Keras: {}".format(cie1976_np(a3, b3), K.eval(cie1976_keras(K.variable(a3), K.variable(b3)))))
    print("Numpy: {} Keras: {}".format(cie1976_np(a3, c3), K.eval(cie1976_keras(K.variable(a3), K.variable(c3)))))

    print("2000:")
    print("Numpy: {} Keras: {}".format(cie2000_np(a3, a3), K.eval(cie2000_keras(K.variable(a3), K.variable(a3)))))
    print("Numpy: {} Keras: {}".format(cie2000_np(a3, b3), K.eval(cie2000_keras(K.variable(a3), K.variable(b3)))))
    print("Numpy: {} Keras: {}".format(cie2000_np(a3, c3), K.eval(cie2000_keras(K.variable(a3), K.variable(c3)))))


# test_delta_e()
