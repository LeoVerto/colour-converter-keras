#!/usr/bin/env python3

from colour_converter import get_model
from keras.utils import plot_model

file = "model.png"

plot_model(get_model(), show_shapes=True, show_layer_names=True, to_file=file)
print("Saved model visualization to " + file)
