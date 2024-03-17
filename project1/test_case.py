
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import warnings


N_EPOCHS = 100


def run_test_case(layers, kernel_size, padding, stride, pooling, normalization, optimizer, train_ds, val_ds, test_ds):

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

    model.build()
    print(model.summary())
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(train_ds, epochs=N_EPOCHS,
              validation_data=val_ds)


def get_1_layer_model(kernel_size, padding, stride, pooling, normalization, optimizer):
    model = models.Sequential()

    # CONVOLUTION LAYER 1
    if padding > 0:
        model.add(layers.ZeroPadding2D(
            padding=(padding, padding), input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (kernel_size, kernel_size),
                            activation='relu', input_shape=(32 + padding, 32 + padding, 3)))
    model = add_pooling_layer(model, pooling)

    # DENSE LAYERS
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def get_3_layer_model(kernel_size, padding, stride, pooling, normalization, optimizer):
    model = models.Sequential()

    # CONVOLUTION LAYER 1
    if padding > 0:
        model.add(layers.ZeroPadding2D(
            padding=(padding, padding), input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (kernel_size, kernel_size),
                            activation='relu', input_shape=(32 + padding, 32 + padding, 3)))
    model = add_pooling_layer(model, pooling)

    # CONVOLUTION LAYER 2
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(64, (kernel_size, kernel_size),
              activation='relu'))
    model = add_pooling_layer(model, pooling)

    # CONVOLUTION LAYER 3
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(64, (kernel_size, kernel_size),
              activation='relu'))
    model = add_pooling_layer(model, pooling)

    # DENSE LAYERS
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10,  activation='softmax'))
    return model


def get_5_layer_model(kernel_size, padding, stride, pooling, normalization, optimizer):
    model = models.Sequential()

    # CONVOLUTION LAYER 1
    if padding > 0:
        model.add(layers.ZeroPadding2D(
            padding=(padding, padding), input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (kernel_size, kernel_size),
                            activation='relu', input_shape=(32 + padding, 32 + padding, 3)))

    # CONVOLUTION LAYER 2
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(32, (kernel_size, kernel_size), activation='relu'))
    model = add_pooling_layer(model, pooling)

    # CONVOLUTION LAYER 3
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(64, (kernel_size, kernel_size), activation='relu'))

    # CONVOLUTION LAYER 4
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(64, (kernel_size, kernel_size), activation='relu'))
    model = add_pooling_layer(model, pooling)

    # CONVOLUTION LAYER 5
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(64, (kernel_size, kernel_size), activation='relu'))

    # DENSE LAYERS
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def add_pooling_layer(model, pooling, pooling_size=2):
    match(pooling):
        case 'max_pooling':
            model.add(layers.MaxPooling2D((pooling_size, pooling_size)))
        case 'average_pooling':
            model.add(layers.AveragePooling2D((pooling_size, pooling_size)))
        case _:
            pass
    return model


def get_next_dimension(input_vol, kernel, padding, stride):
    return (input_vol - kernel + 2*padding)/stride + 1
