#!/usr/bin/env python3

import cv2
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda
from keras.callbacks import TensorBoard, Callback
from skimage import color
from utils import *
from delta_e import *

# Config

BATCH_SIZE = 20
EPOCH = 5
DRAW_WAIT = 5  # Set to 0 to disable drawing

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

training = np.load("training.npy")
firstRun = True


def loss(y_true, y_pred):
    return K.mean(K.square(cie1976_keras(y_true, y_pred)))


def train(model, render=False):
    training_rgb = training[:, [0, 1, 2]];
    training_lab = training[:, [6, 7, 8]];

    if render:
        draw_callback = DrawCallback()
        model.fit(training_rgb, training_lab, nb_epoch=EPOCH, batch_size=BATCH_SIZE,
                  callbacks=[tensorboard, draw_callback])
    else:
        model.fit(training_rgb, training_lab, nb_epoch=EPOCH, batch_size=BATCH_SIZE, callbacks=[tensorboard])


def evaluate(model):
    validation = np.load("validation.npy")

    print("EVALUATION:")

    loss_and_metrics = model.evaluate(validation[:, [0, 1, 2]], validation[:, [6, 7, 8]], batch_size=BATCH_SIZE)
    print(loss_and_metrics)

    draw(model, 0)


def predict(model):
    # RGB: 29.11, 94.53, 198.88
    # LAB: 41.97, 19.45, -59.9

    test_arrays = np.array([
        [[0.11414414, 0.37071942, 0.77993838]],
        [[1, 0, 0]],
        [[0, 1, 0]],
        [[0, 0, 1]]
    ])

    print("PREDICTIONS:")

    for array in test_arrays:
        print(rgb2string(array[0]))
        print("Expected:")
        print(lab2string(pixel_rgb2lab(array[0])))
        print("Got:")
        print(lab2string(model.predict(array)[0]))


def draw(model, wait):
    data = training

    img = np.copy(data[:10, [0, 1, 2]])
    img = np.reshape(img, (10, 1, 3))

    b = data[:10, [0, 1, 2]]
    predictions = np.divide(np.reshape(model.predict(b), (10, 1, 3)), 128)
    pred_rgb = np.multiply(color.lab2rgb(predictions), 64)  # Magic 64, no idea why this needs to be multiplied
    img = np.append(img, pred_rgb, axis=1)
    img = np.reshape(img, (10, 2, 3))

    cv2.namedWindow("Keras Colour Converter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Keras Colour Converter", 600, 600)
    cv2.imshow("Keras Colour Converter", img)

    global firstRun
    if firstRun:
        cv2.waitKey(0)
        firstRun = False

    key = cv2.waitKey(wait)

    if key == ord('q'):
        sys.exit("User terminated program.")


class DrawCallback(Callback):
    def on_batch_end(self, batch, logs={}):
        draw(self.model, DRAW_WAIT)


def run():
    model = Sequential([
        Dense(12, input_dim=3),
        Activation("relu"),
        Dense(18),
        Activation("relu"),
        Dense(3),
        Activation("linear"),
        Lambda(lambda x: x * 128)  # Multiply by 128 as a* and b* may be negative up to -128
    ])

    model.compile(loss=loss, optimizer="adam", metrics=["accuracy", cie1976_keras])

    render = DRAW_WAIT != 0

    train(model, render)
    evaluate(model)
    predict(model)


# test_delta_e()
run()
# print(training[:10])
