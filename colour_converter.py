#!/usr/bin/env python3

import cv2
import sys
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Input, Lambda
from keras.callbacks import TensorBoard, Callback
from utils import *
from delta_e import cie1976_keras, cie2000_keras

# Config

BATCH_SIZE = 20
EPOCHS = 5
DRAW_WAIT = -1  # Set to -1 to disable drawing
DRAW_EVERY = 10  # Only draw every n batches to speed up training

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

training = np.load("training.npy")
firstRun = True


def loss(y_true, y_pred):
    return K.mean(K.square(cie2000_keras(y_true, y_pred)))


def train(model, render=False):
    training_rgb = training[:, [0, 1, 2]]
    training_lab = training[:, [6, 7, 8]]

    callbacks = [tensorboard]

    if render:
        draw_callback = DrawCallback()
        callbacks.append(draw_callback)

    model.fit(training_rgb, training_lab, epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=callbacks, validation_split=0.1)


def evaluate(model):
    evaluation = np.load("evaluation.npy")

    print("EVALUATION:")

    loss_and_metrics = model.evaluate(evaluation[:, [0, 1, 2]], evaluation[:, [6, 7, 8]], batch_size=BATCH_SIZE)
    print(loss_and_metrics)


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
        print("Expected:" + lab2string(pixel_rgb2lab(array[0])))
        print("Got:     " + lab2string(model.predict(array)[0]))


def draw(model, wait):
    data = training

    img = np.copy(data[:10, [0, 1, 2]]).reshape((10, 1, 3))

    predictions = model.predict(data[:10, [0, 1, 2]]).reshape([10, 1, 3])
    #print(predictions)

    predictions = np.subtract(predictions, [50, 0, 0])
    predictions = np.divide(predictions, [50, 128, 128])
    #print(predictions)

    pred_rgb = color.lab2rgb(predictions)

    pred_rgb = np.multiply(pred_rgb, 128)

    img = np.append(img, pred_rgb, axis=1).reshape((10, 2, 3))

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
    def __init__(self):
        super().__init__()
        self.batch = 0
        self.N = DRAW_EVERY

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            self.batch = 0
            draw(self.model, DRAW_WAIT)
        self.batch += 1


def get_model():
    inputs = Input(shape=(3,), name="Input")
    dense1 = Dense(12, activation="relu", name="Relu_Dense")(inputs)
    dense2 = Dense(18, activation="relu", name="Relu_Dense_2")(dense1)
    dense3 = Dense(3, activation="linear", name="Linear_Dense")(dense2)

    mul = K.constant(np.array([50, 128, 128]))
    add = K.constant(np.array([50, 0, 0]))

    scaler = Lambda(lambda x: x * mul + add, name="Output_Scaler")(dense3)

    model = Model(inputs=inputs, outputs=scaler)

    return model


def run():
    model = get_model()
    model.compile(loss=loss, optimizer="adam", metrics=[cie1976_keras, cie2000_keras])

    render = DRAW_WAIT != -1

    #train(model, render)
    model = load_model("model.h5")
    #evaluate(model)
    #predict(model)
    model.save("model.h5")
    draw(model, 0)


if __name__ == '__main__':
    run()
