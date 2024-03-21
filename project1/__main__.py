# supress PNG warning: iCCP: known incorrect sRGB profile
# must be set before import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import logging
from dataset_preparation import *
from test_case import run_test_case


layers = [5]
kernel_sizes = [5]
padding = [0, 1, 2]
stride = [1, 2, 3, 5]
pooling = [None, 'max_pooling', 'average_pooling']
normalization = [None, 'batch', 'layer']
optimizers = [None, 'adam', 'SGD', 'rmsprop', 'adagrad']
seeds = [1, 4, 89, 901, 2137]


train_ds = get_train_data(128)
val_ds = get_validation_data()
test_ds = get_test_data()

# if using stride, add padding of floor(stride/2)
# if using pooling add padding  >= 1
for seed in seeds:           
    for layer in layers:
        for kernel in kernel_sizes:
            print("Testing for: \nlayers: " + str(layer) +
                '\nkernel: ' + str(kernel) + '\nseed: ' + str(seed))
            save_path = "layer-" + str(layer) + '__kernel-' + str(kernel)
            run_test_case(layers=layer, kernel_size=kernel, padding=2, stride=5, pooling=None,
                        normalization='batch', optimizer='adam', train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, path=save_path, seed=seed)
