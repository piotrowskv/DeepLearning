# supress PNG warning: iCCP: known incorrect sRGB profile
# must be set before import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import logging
from dataset_preparation import *
from test_case import run_test_case


layers = [3, 5]
kernel_sizes = [3, 5]
padding = [0, 1, 2]
stride = [1, 2, 3, 5]  # TODO
pooling = [None, 'max_pooling', 'average_pooling']
normalization = [None, 'batch', 'layer']
optimizers = [None, 'adam', 'SGD', 'rmsprop', 'adagrad']
seeds = [1, 4, 89, 901, 2137]


train_ds = get_train_data(128)
val_ds = get_validation_data()
test_ds = get_test_data()
                  
for layer in layers:
    for kernel in kernel_sizes:
        if layer == 1 and kernel == 3:
            continue
        for seed in seeds:
            print("Testing for: \nlayers: " + str(layer) +
                  '\nkernel: ' + str(kernel) + '\nseed: ' + str(seed))
            save_path = "layer-" + str(layer) + '__kernel-' + str(kernel)
            run_test_case(layers=layer, kernel_size=kernel, padding=0, stride=1, pooling='max_pooling',
                          normalization='batch', optimizer='adam', train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, path=save_path, seed=seed)
