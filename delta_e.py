import numpy as np
from keras import backend as K
from colormath.color_diff_matrix import delta_e_cie2000


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
    # y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    distance = K.sqrt(
        K.square(y_pred[:, 0]-y_true[:, 0])
        + K.square(y_pred[:, 1]-y_true[:, 1])
        + K.square(y_pred[:, 2]-y_true[:, 2])
    )
    return distance


# Delta E 2000 using NumPy
def cie2000_np(y_true, y_pred):
    return delta_e_cie2000(K.eval(y_true), K.eval(y_pred))


# Delta E 2000 using Keras
# https://github.com/gtaylor/python-colormath/blob/master/colormath/color_diff_matrix.py#L112
# noinspection PyPep8Naming
def cie2000_keras(y_true, y_pred, Kl=1, Kc=1, Kh=1):
    L, a, b = y_true

    avg_Lp = (L + y_pred[:, 0]) / 2.0

    C1 = K.sqrt(K.sum(K.pow(y_true[1:], 2)))
    C2 = K.sqrt(K.sum(K.pow(y_pred[:, 1:], 2), axis=1))

    avg_C1_C2 = (C1 + C2) / 2.0

    G = 0.5 * (1 - K.sqrt(K.pow(avg_C1_C2, 7.0) / (K.pow(avg_C1_C2, 7.0) + K.pow(25.0, 7.0))))

    a1p = (1.0 + G) * a
    a2p = (1.0 + G) * y_pred[:, 1]

    C1p = K.sqrt(K.pow(a1p, 2) + K.pow(b, 2))
    C2p = K.sqrt(K.pow(a2p, 2) + K.pow(y_pred[:, 2], 2))

    avg_C1p_C2p = (C1p + C2p) / 2.0

    h1p = K.degrees(K.arctan2(b, a1p))
    h1p += (h1p < 0) * 360

    h2p = K.degrees(K.arctan2(y_pred[:, 2], a2p))
    h2p += (h2p < 0) * 360

    avg_Hp = (((K.fabs(h1p - h2p) > 180) * 360) + h1p + h2p) / 2.0

    T = 1 - 0.17 * K.cos(K.radians(avg_Hp - 30)) + \
        0.24 * K.cos(K.radians(2 * avg_Hp)) + \
        0.32 * K.cos(K.radians(3 * avg_Hp + 6)) - \
        0.2 * K.cos(K.radians(4 * avg_Hp - 63))

    diff_h2p_h1p = h2p - h1p
    delta_hp = diff_h2p_h1p + (K.fabs(diff_h2p_h1p) > 180) * 360
    delta_hp -= (h2p > h1p) * 720

    delta_Lp = y_pred[:, 0] - L
    delta_Cp = C2p - C1p
    delta_Hp = 2 * K.sqrt(C2p * C1p) * K.sin(K.radians(delta_hp) / 2.0)

    S_L = 1 + ((0.015 * K.pow(avg_Lp - 50, 2)) / K.sqrt(20 + K.pow(avg_Lp - 50, 2.0)))
    S_C = 1 + 0.045 * avg_C1p_C2p
    S_H = 1 + 0.015 * avg_C1p_C2p * T

    delta_ro = 30 * K.exp(-(K.pow(((avg_Hp - 275) / 25), 2.0)))
    R_C = K.sqrt((K.pow(avg_C1p_C2p, 7.0)) / (K.pow(avg_C1p_C2p, 7.0) + K.pow(25.0, 7.0)))
    R_T = -2 * R_C * K.sin(2 * K.radians(delta_ro))

    return K.sqrt(
        K.pow(delta_Lp / (S_L * Kl), 2) +
        K.pow(delta_Cp / (S_C * Kc), 2) +
        K.pow(delta_Hp / (S_H * Kh), 2) +
        R_T * (delta_Cp / (S_C * Kc)) * (delta_Hp / (S_H * Kh)))


def test_delta_e():
    a = np.array([47, 26, -23])
    b = np.array([47, 26, -24])
    c = np.array([58, -28, 18])

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
