# supress PNG warning: iCCP: known incorrect sRGB profile
# must be set before import tensorflow
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from test_case import run_test_case
from dataset_preparation import *
import logging
import tensorflow as tf



layers = [1, 3, 5]
kernel_sizes = [3, 5]
padding = [0, 1, 2] 
stride = [1, 2, 3, 5]  # TODO
pooling = [None, 'max_pooling', 'average_pooling']
normalization = [None, 'batch', 'layer']  # TODO
optimizeers = [None, 'adam', 'SGD', 'rmsprop', 'adagrad']

train_ds = get_train_data(128)
val_ds = get_validation_data()
#test_ds = get_test_data()
run_test_case(layers=3, kernel_size=3, padding=0, stride=1, pooling='max_pooling',
              normalization=None, optimizer='adam', train_ds=train_ds, val_ds=val_ds, test_ds=None)
