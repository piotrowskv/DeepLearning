
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import warnings
from results_utils import *
from keras.utils import to_categorical
import numpy as np

N_EPOCHS = 1


def run_test_case(layers, kernel_size, padding, stride, pooling, normalization, optimizer, train_ds, val_ds, test_ds, path, seed):
    tf.random.set_seed(seed)
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
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=["accuracy"])
    history = model.fit(train_ds, epochs=N_EPOCHS,
                        validation_data=val_ds)
    # predictions = model.predict(test_ds)
    # predictions = np.argmax(predictions, axis=1)
    # print(predictions)
    # predictions = to_categorical(predictions, num_classes=10)
   # labels = train_ds.class_names
   # print(predictions)
    # y = np.concatenate([y for x, y in test_ds], axis=0)
    # y = np.argmax(y, axis=1)
  #  print(y)
   # predictions = np.argmax(predictions, axis=1)
    result = model.evaluate(test_ds)
    result_dict = (dict(zip(model.metrics_names, result)))

    # correct_predictions = np.sum(np.where(y==predictions, 1, 0))
    # accuracy_test = correct_predictions / len(predictions)
   # plot_confusion_matrix(predictions, y, labels, path + '/seed-' + str(seed))
    plot_accuracy(history, path + '/seed-' + str(seed))
    plot_loss(history, path + '/seed-' + str(seed))
    append_accuracy_score(result_dict['accuracy'], path)


def get_1_layer_model(kernel_size, padding, stride, pooling, normalization, optimizer):
    model = models.Sequential()

    # CONVOLUTION LAYER 1
    if padding > 0:
        model.add(layers.ZeroPadding2D(
            padding=(padding, padding), input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(16, (kernel_size, kernel_size), strides=(stride, stride),
                            activation='relu', input_shape=(32 + padding, 32 + padding, 3)))
    model = add_pooling_layer(model, pooling)
    model = add_normalization_layer(model, normalization)

    # DENSE LAYERS
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def get_3_layer_model(kernel_size, padding, stride, pooling, normalization, optimizer):
    model = models.Sequential()

    # CONVOLUTION LAYER 1
    if padding > 0:
        model.add(layers.ZeroPadding2D(
            padding=(padding, padding), input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(16, (kernel_size, kernel_size), strides=(stride, stride),
                            activation='relu', input_shape=(32 + padding, 32 + padding, 3)))
    model = add_pooling_layer(model, pooling)
    model = add_normalization_layer(model, normalization)

    # CONVOLUTION LAYER 2
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(16, (kernel_size, kernel_size), strides=(stride, stride),
              activation='relu'))
    model = add_normalization_layer(model, normalization)

    # CONVOLUTION LAYER 3
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(16, (kernel_size, kernel_size), strides=(stride, stride),
              activation='relu'))
    model = add_pooling_layer(model, pooling)
    model = add_normalization_layer(model, normalization)

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
    model.add(layers.Conv2D(32, (kernel_size, kernel_size), strides=(stride, stride),
                            activation='relu', input_shape=(32 + padding, 32 + padding, 3)))
    model = add_model = add_pooling_layer(model, pooling)
    model = add_normalization_layer(model, normalization)

    # CONVOLUTION LAYER 2
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(16, (kernel_size, kernel_size),
              activation='relu'))
    # model = add_pooling_layer(model, pooling)
    model = add_normalization_layer(model, normalization)

    # CONVOLUTION LAYER 3
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(32, (kernel_size, kernel_size),
               activation='relu'))
    model = add_normalization_layer(model, normalization)

    # CONVOLUTION LAYER 4
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(32, (kernel_size, kernel_size),
               activation='relu'))
    model = add_normalization_layer(model, normalization)

    # CONVOLUTION LAYER 5
    if padding > 0:
        model.add(layers.ZeroPadding2D(padding=(padding, padding)))
    model.add(layers.Conv2D(32, (kernel_size, kernel_size),
              strides=(stride, stride), activation='relu'))
    model = add_pooling_layer(model, pooling)
    model = add_normalization_layer(model, normalization)

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


def add_normalization_layer(model, normalization):
    match(normalization):
        case 'batch':
            model.add(layers.BatchNormalization())
        case 'layer':
            model.add(layers.LayerNormalization())
        case _:
            pass
    return model


def get_next_dimension(input_vol, kernel, padding, stride):
    return (input_vol - kernel + 2*padding)/stride + 1
