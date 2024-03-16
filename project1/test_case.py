
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from dataset_preparation import *
import warnings


N_EPOCHS = 100


def run_test_case(layers, kernel_size, padding, stride, pooling, normalization, optimizer):

    match(layers):
        case 1:
            model = get_1_layer_model(
                kernel_size, padding, stride, pooling, normalization, optimizer)
        case 3:
            model = get_3_layer_model(
                kernel_size, padding, stride, pooling, normalization, optimizer)

        case 5:
            model = get_5_layer_model(
                kernel_size, padding, stride, pooling, normalization, optimizer)

    print(model.summary())
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    model.fit(get_train_data(128), epochs=N_EPOCHS,
              validation_data=get_validation_data())


def get_1_layer_model(kernel_size, padding, stride, pooling, normalization, optimizer):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (kernel_size, kernel_size),
                            activation='relu', input_shape=(32, 32, 3)))
    model = add_pooling_layer(model, pooling)

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10))

    return model


def get_3_layer_model(kernel_size, padding, stride, pooling, normalization, optimizer):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (kernel_size, kernel_size),
                            activation='relu', input_shape=(32, 32, 3)))
    model = add_pooling_layer(model, pooling)

    model.add(layers.Conv2D(64, (kernel_size, kernel_size), activation='relu'))
    model = add_pooling_layer(model, pooling)

    model.add(layers.Conv2D(64, (kernel_size, kernel_size), activation='relu'))
    model = add_pooling_layer(model, pooling)

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model


def get_5_layer_model(kernel_size, padding, stride, pooling, normalization, optimizer):

    model = models.Sequential()
    model.add(layers.Conv2D(32, (kernel_size, kernel_size),
                            activation='relu', input_shape=(32, 32, 3)))
    model = add_pooling_layer(model, pooling)

    model.add(layers.Conv2D(64, (kernel_size, kernel_size), activation='relu'))
    model = add_pooling_layer(model, pooling)

    model.add(layers.Conv2D(64, (kernel_size, kernel_size), activation='relu'))
    add_pooling_layer(model, pooling)

    model.add(layers.Conv2D(64, (kernel_size, kernel_size), activation='relu'))
    add_pooling_layer(model, pooling)

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model


def add_pooling_layer(model, pooling):
    match(pooling):
        case 'max_pooling':
            model.add(layers.MaxPooling2D((2, 2)))
        case 'average_pooling':
            model.add(layers.AveragePooling2D((2, 2)))
    return model
