# Keras Colour Converter
A machine learning model that can learn to convert colours 
from the RGB space to L\*a\*b\* using 

This project was inspired by [jeyj0/PyNNet](https://github.com/jeyj0/PyNNet).

I do realize this may not be a very interesting application for machine learning
but it seemed appropriate to get familiar with Keras and only took the better
part of a night to build.

I chose the L\*a\*b\* colour space as the output, because it easily allows 
for calculation [Delta E](https://en.wikipedia.org/wiki/Color_difference#LAB_Delta_E),
which is a pretty consistent measurement of how humans perceive colour difference.

The formula used for the loss function is CIE76 but other ones can easily be implemented.

After training, the model should be able to reach a Delta E of around 3.3 so there is
definitely room for further improvement.

## Running
1. Run `generate_dataset.py` to generate a training and validation set.
2. Run `colour_converter.py` to train and test the model.