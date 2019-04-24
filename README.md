# Keras Colour Converter
A machine learning model that can learn to convert colours 
from the RGB space to L\*a\*b\* using 

This project was inspired by [jeyj0/PyNNet](https://github.com/jeyj0/PyNNet).

I do realize this may not be a very interesting application for machine learning
but it seemed appropriate to get familiar with Keras and only took the better
part of ~~a night~~ two nights to build.

I chose the L\*a\*b\* colour space as the output because it easily allows 
for calculation [Delta E](https://zschuessler.github.io/DeltaE/learn/),
which is a pretty consistent measurement of how humans perceive colour difference.

The formula used for the loss function is the color difference function CIEDE2000 (aka Delta E 2000)
but other ones can easily be implemented.

The model now consistently reaches a Delta E of around 1.3, which should be almost
imperceptible to the human eye.


## Running
1. Run `generate_dataset.py` to generate a training and validation set.
2. Run `colour_converter.py` to train and evaluate the model.
3. If rendering is enabled, press any key to start training.

## Findings
Disclaimer: None of these very properly verified by more than one test run.

* SGD doesn't really work, most training attempts wouldn't even remotely
resemble any optimization
* Adam however works quite well
* Squared mean loss compared to mean loss trades off initial training speed
for better accuracy in the long run
* The output layer should be activated linearly (translates well to LAB output range)
* The earlier layers prefer being activated by relu (RGB inputs are only positive)
* A batch size of 20 works pretty well, larger ones not so much
* Adding another hidden layer improved accuracy a lot and so did adding more
neurons to it