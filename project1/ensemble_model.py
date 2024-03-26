# supress PNG warning: iCCP: known incorrect sRGB profile
# must be set before import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dataset_preparation import *
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import warnings
from results_utils import *
from keras.utils import to_categorical
import numpy as np
N_EPOCHS = 20
N_EPOCHS_ENSEMBLE = 1

def prepare_dataset_for_class(chosen_class, dataset):
    chosen_ds = dataset.unbatch().filter(lambda x, y:  tf.equal(y, chosen_class))
    other_classes = dataset.unbatch().filter(
        lambda x, y:  tf.not_equal(y, chosen_class))
    chosen_ds = chosen_ds.map(lambda img, box: (img, 1))
    other_classes = other_classes.shuffle(buffer_size=3)
    other_classes = other_classes.take(9000)
    other_classes = other_classes.map(lambda img, box: (img, 0))
    concat_ds = other_classes.concatenate(chosen_ds)
    concat_ds = concat_ds.map(lambda x, y: (x, y)).batch(batch_size=128)
    return concat_ds.shuffle(buffer_size=3)


def prepare_model():
    model = models.Sequential()
    padding = 2
    kernel_size = 3
    stride = 2

    model.add(layers.ZeroPadding2D(
        padding=(padding, padding), input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(16, (kernel_size, kernel_size), strides=(stride, stride),
                            activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    # DENSE LAYERS
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def train_one_model(class_name):
    print("Training for " + class_name)
    idx = train_ds.class_names.index(class_name)
    train_ds_for_class = prepare_dataset_for_class(idx, train_ds)
    val_ds_for_class = prepare_dataset_for_class(idx, val_ds)
    model = prepare_model()
    model.build()
    callback = callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=["accuracy"])
    history = model.fit(train_ds_for_class, validation_data=val_ds_for_class, epochs=N_EPOCHS, callbacks=[callback])
    return model

path = 'ensemble'
seeds = [4, 89, 901, 2137]

for seed in seeds:
    tf.random.set_seed(seed)
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    trained_models =[]

    train_ds = get_train_data(128, categorical=False)
    val_ds = get_validation_data(categorical=False)
    test_ds = get_test_data(categorical=False)
    for class_name in class_names:
        trained_models.append(train_one_model(class_name))


    train_ds = get_train_data(128)
    val_ds = get_validation_data()
    test_ds = get_test_data()
    input = tf.keras.Input(shape=(32, 32, 3), name='input')
    outputs = [model(input) for model in trained_models]
    x = layers.Concatenate()(outputs)
    output = layers.Dense(10, activation='softmax', name='output')(x)
    ensemble_model = tf.keras.Model(input, output)
    ensemble_model.compile(optimizer='adam',
                    loss='categorical_crossentropy', metrics=["accuracy"])
    history = ensemble_model.fit(train_ds, epochs=N_EPOCHS_ENSEMBLE,
                            validation_data=val_ds)
    predictions = ensemble_model.predict(test_ds)
    predictions = np.argmax(predictions, axis=1)
    y = np.concatenate([y for x, y in test_ds], axis=0)
    y = np.argmax(y, axis=1)
    labels = train_ds.class_names

    result = ensemble_model.evaluate(test_ds)
    result_dict = (dict(zip(ensemble_model.metrics_names, result)))

    plot_confusion_matrix(predictions, y, labels, path + '/seed-' + str(seed))
    plot_accuracy(history, path + '/seed-' + str(seed))
    plot_loss(history, path + '/seed-' + str(seed))
    append_accuracy_score(result_dict['accuracy'], path)
