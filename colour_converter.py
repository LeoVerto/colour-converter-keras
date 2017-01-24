#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda
from keras import backend as K
from utils import rgb2string, lab2string

_EPSILON = K.epsilon()


def delta_e_np(y_true, y_pred):
    distance = np.sqrt(
        np.square(y_pred[:,0]-y_true[:,0])
        + np.square(y_pred[:,1]-y_true[:,1])
        + np.square(y_pred[:,2]-y_true[:,2])
    )
    # print(distance)
    return np.mean(distance)


def delta_e_tensor(y_true, y_pred):
    # y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    distance = K.sqrt(
        K.square(y_pred[:,0]-y_true[:,0])
        + K.square(y_pred[:,1]-y_true[:,1])
        + K.square(y_pred[:,2]-y_true[:,2])
    )
    return K.mean(distance)


def test_delta_e():
    a = np.array([47, 26, -23])
    b = np.array([47, 26, -24])
    c = np.array([58, -28, 18])

    a3 = np.array([a, a, a])
    b3 = np.array([b, b, b])
    c3 = np.array([c, c, c])

    print("Numpy: {} Keras: {}".format(delta_e_np(a3, a3), K.eval(delta_e_tensor(K.variable(a3), K.variable(a3)))))
    print("Numpy: {} Keras: {}".format(delta_e_np(a3, b3), K.eval(delta_e_tensor(K.variable(a3), K.variable(b3)))))
    print("Numpy: {} Keras: {}".format(delta_e_np(a3, c3), K.eval(delta_e_tensor(K.variable(a3), K.variable(c3)))))


def train(model):
    training = np.load("training.npy")

    model.fit(training[:, [0,1,2]], training[:, [6,7,8]], nb_epoch=5, batch_size=15)


def evaluate(model):
    validation = np.load("validation.npy")

    print("EVALUATION:")

    loss_and_metrics = model.evaluate(validation[:, [0,1,2]], validation[:, [6,7,8]], batch_size=10)
    print(loss_and_metrics)


def predict(model):
    # RGB: 29.11, 94.53, 198.88
    # LAB: 41.97, 19.45, -59.9

    test_arrays= np.array([
        [[0.11414414, 0.37071942, 0.77993838]],
        [[1, 0, 0]],
        [[0, 1, 0]],
        [[0, 0, 1]]
    ])

    print("PREDICTIONS:")

    for array in test_arrays:
        print(rgb2string(array[0]))
        print(lab2string(model.predict(array)[0]))


def run():
    model = Sequential([
        Dense(9, input_dim=3),
        Activation("relu"),
        Dense(3),
        Activation("relu"),
        Lambda(lambda x: x - 128)  # Magic lambda, a* and b* may be negative up to -128
    ])

    model.compile(loss=delta_e_tensor, optimizer="sgd", metrics=["accuracy"])

    train(model)
    evaluate(model)
    predict(model)

# test_delta_e()
run()
