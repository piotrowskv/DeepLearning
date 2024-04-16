import tensorflow
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from python_speech_features import mfcc
import os
import malaya_speech
from malaya_speech import Pipeline
import noisereduce as nr
from scipy.io import wavfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import warnings
warnings.filterwarnings("ignore")


sr = 16000  # sampling rate
vad = malaya_speech.vad.webrtc()
train_audio_path = 'data/train/audio'
labels = ["bed", "bird", "cat", "dog", "down", "go", "happy", "house",
          "left", "marvin", "no", "off", "on", "right", "sheila", "stop", "tree",
          "up", "wow", "yes", "zero", "one", "two", "three", "four",
          "five", "six", "seven", "eight", "nine", "other"]
labels_other = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]

def load_from_file(filename):
    all_wave = []
    all_label = []
    with open(filename, 'r+') as f:
        for wav in f:
            label = get_label(wav)
            wav = wav.replace('\n', '')
            if label in labels_other:
                label = "other"
            input_length = 16000
            data = librosa.core.load(
                train_audio_path + '/' + wav)[0]  # , sr = 16000

            if len(data) > input_length:
                data = data[: input_length]
            else:
                data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

            ## data cleaning by https://www.kaggle.com/code/mayarmohsen/numbers
            # samples, sample_rate = librosa.load(train_audio_path + '/' + wav, sr=16000)
            # samples = nr.reduce_noise(y=samples, sr=sr, stationary=True)
            # y_ = malaya_speech.resample(samples, sr, 16000)
            # y_ = malaya_speech.astype.float_to_int(y_)
            # frames = malaya_speech.generator.frames(samples, 30, sr)
            # frames_ = list(malaya_speech.generator.frames(
            #     y_, 30, 16000, append_ending_trail=False))
            # frames_webrtc = [(frames[no], vad(frame))
            #                  for no, frame in enumerate(frames_)]
            # y_ = malaya_speech.combine.without_silent(frames_webrtc)
            # zero = np.zeros(((1*sr+4000)-y_.shape[0]))
            # signal = np.concatenate((y_, zero))
            #all_wave.append(signal)
            all_wave.append(data)
            all_label.append(label)
        print("Loaded " + str(np.array(all_wave).shape[0]) + " files!")
        return np.array(all_wave), np.array(all_label)


def get_label(filename):
    return filename.split('/')[0]


def get_train_filenames():
    '''Gets filenames that are not in testing_list and validation_list'''
    for label in labels:
        print(label)
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

def load_all_data():
    test_wav, test_label = load_from_file('data/train/testing_list.txt')
    val_wav, val_label = load_from_file('data/train/validation_list.txt')
    train_wav, train_label = load_from_file(
        'data/train/training_list.txt')
    return test_wav, test_label, val_wav, val_label, train_wav, train_label


load_all_data()