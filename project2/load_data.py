import tensorflow as tf
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
import time
import os
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import warnings
import random
from scipy import signal
from scipy.io import wavfile
warnings.filterwarnings("ignore")


sr = 16000  # sampling rate
train_audio_path = 'data/train/audio'
targets = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
#targets = ["yes", "no", "up"]

def create_spectogram(filepath):
    if not os.path.isfile(filepath):
        return
    audio = tf.io.read_file(filepath)
    audio, _ = tf.audio.decode_wav(audio)

    audio = tf.squeeze(audio, axis=-1)
    sample_rate = 16000


    nperseg = int(round(20 * sample_rate / 1e3))
    noverlap = int(round(10 * sample_rate / 1e3))
    # stfts = tf.signal.stft(audio, 400, 160)
    freqs, times, x = signal.spectrogram(audio,
                   fs=sample_rate,
                   window='hann',
                   nperseg=nperseg,
                   noverlap=noverlap,
                   detrend=False)
    #x = tf.math.pow(tf.abs(x), 0.5)
    x = tf.math.log(x.T.astype(np.float32) + 1e-3)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / (stddevs + 1e-6)
    audio_len = tf.shape(x).numpy()[0]

    file_length = 99
    if(audio_len < file_length):
        padding_size = file_length - audio_len
        paddings = tf.constant([[0, padding_size], [0, 0]])
        x = tf.pad(x, paddings, "CONSTANT")
    if(audio_len > file_length):
        x = x[:file_length, :]


    return x

def log_specgram(audio, sample_rate=16000, window_size=20,
                 step_size=10, eps=1e-10):
    sample_rate, samples = wavfile.read(audio)
    audio_len = tf.shape(samples).numpy()[0]

    file_length = 16000
    if(audio_len < file_length):
        padding_size = file_length - audio_len
        paddings = tf.constant([[0, padding_size], [0, 0]])
        samples = tf.pad(samples, paddings, "CONSTANT")
    if(audio_len > file_length):
        samples = samples[:file_length, :]
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(samples,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return np.log(spec.T.astype(np.float32) + 1e-3)

def load_from_file(filename):
    all_wave = []
    all_label = []
    tf_waves = []
    
    with open(filename, 'r+') as f:
        for wav in f:
            label = get_label(wav)
            wav = wav.replace('\n', '')
            if label=='_background_noise_':
                continue
                #label = 'silence'
            elif label not in targets:
                continue
                #label = "unknown"
            path = train_audio_path + '/' + wav
            all_wave.append(path)
            all_label.append(targets.index(label))

    shuffled = list(zip(all_wave, all_label))
    random.shuffle(shuffled)
    all_wave, all_label = zip(*shuffled)

    tf_waves = list(map(create_spectogram, all_wave))

    tf_waves = tf.data.Dataset.from_tensor_slices(tf_waves)
    print("Loaded " + str(np.array(all_label).shape[0]) + " files!")
    all_label = tf.data.Dataset.from_tensor_slices(tf.one_hot(all_label, len(targets)).numpy())
    return tf_waves, all_label


def get_label(filename):
    return filename.split('/')[0]


def get_train_filenames():
    '''Gets filenames that are not in testing_list and validation_list'''
    labels = ["bed", "bird", "cat", "dog", "down", "go", "happy", "house",
              "left", "marvin", "no", "off", "on", "right", "sheila", "stop", "tree",
              "up", "wow", "yes", "zero", "one", "two", "three", "four",
              "five", "six", "seven", "eight", "nine"]
    for label in labels:
        print("Reading files for: " + label)
        for root, dirs, files in os.walk(train_audio_path + '/' + label):
            for file in files:
                if file.endswith(".wav"):
                    with open('data/train/testing_list.txt', 'r+') as f:
                        for line in f:
                            if (label + '/' + file) in line:
                                break
                        else:  # not found
                            with open('data/train/validation_list.txt', 'r+') as f2:
                                for line in f2:
                                    if (label + '/' + file) in line:
                                        break
                                else:
                                    with open('training_list.txt', 'a') as f3:
                                        f3.write(label + '/' + file + '\n')

def create_tf_dataset(data, targets, bs=32):
    ds = tf.data.Dataset.zip((data, targets))
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def load_all_data():

    test_wav, test_label = load_from_file('data/train/testing_list.txt')
    test_ds = create_tf_dataset(test_wav, test_label)


    val_wav, val_label = load_from_file('data/train/validation_list.txt')
    val_ds = create_tf_dataset(val_wav, val_label)
    train_wav, train_label = load_from_file(
        'data/train/training_list.txt')
    train_ds = create_tf_dataset(train_wav, train_label)
    return test_ds, val_ds, train_ds


