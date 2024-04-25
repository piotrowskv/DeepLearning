import keras
import numpy as np
from keras.layers import Dense, Reshape, Flatten, LayerNormalization, BatchNormalization, LSTM
from tensorflow.keras import callbacks
import tensorflow as tf
from transformer import TransformerEncoder, SpeechFeatureEmbedding
from load_data import load_all_data
from results_utils import *

#labels = ["yes", "no", "up"]
labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

NUM_CLASSES=10
def train_transformer(test_ds, val_ds, train_ds, embedding=True, embed_dim=8, num_heads=2, ff_dim=32, path='result-sample', seed=3):
    tf.random.set_seed(seed)

    model = keras.Sequential()
    model.add(BatchNormalization())
    if embedding:
        model.add(SpeechFeatureEmbedding(num_hid=embed_dim))
        model.add(BatchNormalization())
    model.add(TransformerEncoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        feed_forward_dim=ff_dim,
        rate=0.1
    ))
    model.add(Flatten())
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    loss_fn = keras.losses.CategoricalCrossentropy(
    )
    learning_rate = 0.2
    model.build(input_shape=((None, 98, 257)))

    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn,  metrics=['accuracy'])
    callback = callbacks.EarlyStopping(monitor='accuracy', patience=3)
    history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[callback])

    predictions = model.predict(test_ds)
    predictions = np.argmax(predictions, axis=1)
    y = np.concatenate([y for x, y in test_ds], axis=0)
    y = np.argmax(y, axis=1)
    
    result = model.evaluate(test_ds)
    result_dict = (dict(zip(model.metrics_names, result)))

    plot_confusion_matrix(predictions, y, labels, path + '/seed-' + str(seed))
    plot_accuracy(history, path + '/seed-' + str(seed))
    plot_loss(history, path + '/seed-' + str(seed))
    append_accuracy_score(result_dict['accuracy'], path)

def train_lstm(test_ds, val_ds, train_ds, n_lstm=2, embedding=True, embed_dim=8, recurrent_dropout=0, path='result-sample', seed=3):
    tf.random.set_seed(seed)
    model = keras.Sequential()
    model.add(BatchNormalization())
    if embedding:
        model.add(SpeechFeatureEmbedding(num_hid=embed_dim))
        model.add(BatchNormalization())

    model.add(LSTM(n_lstm, recurrent_dropout=recurrent_dropout))

    model.add(Flatten())
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    loss_fn = keras.losses.CategoricalCrossentropy(
    )
    learning_rate = 0.2
    model.build(input_shape=((None, 98, 257)))

    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn,  metrics=['accuracy'])
    callback = callbacks.EarlyStopping(monitor='accuracy', patience=3)

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=100, callbacks=[callback])

    predictions = model.predict(test_ds)
    predictions = np.argmax(predictions, axis=1)
    y = np.concatenate([y for x, y in test_ds], axis=0)
    y = np.argmax(y, axis=1)
    result = model.evaluate(test_ds)
    result_dict = (dict(zip(model.metrics_names, result)))

    plot_confusion_matrix(predictions, y, labels, path + '/seed-' + str(seed))
    plot_accuracy(history, path + '/seed-' + str(seed))
    plot_loss(history, path + '/seed-' + str(seed))
    append_accuracy_score(result_dict['accuracy'], path)