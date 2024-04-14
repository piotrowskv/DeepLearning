# by https://www.kaggle.com/code/mayarmohsen/numbers
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
import tensorflow


sr = 16000  # sampling rate
vad = malaya_speech.vad.webrtc()
train_audio_path = 'data/train/audio'
labels = ["zero", "one", "two", "three", "four",
          "five", "six", "seven", "eight", "nine", "other"]

def load_files():
    all_wave = []
    all_label = []
    for label in labels:
        print("Loading " + label + "...")
        waves = []
        for root, dirs, files in os.walk(train_audio_path + '/' + label):
            for file in files:
                if file.endswith(".wav"):
                    waves.append(os.path.join(root, file))
        for wav in waves:
            wav = wav.replace("\\", '/')
            samples, sample_rate = librosa.load(wav, sr=16000)
            samples = nr.reduce_noise(y=samples, sr=sr, stationary=True)
            y_ = malaya_speech.resample(samples, sr, 16000)
            y_ = malaya_speech.astype.float_to_int(y_)
            frames = malaya_speech.generator.frames(samples, 30, sr)
            frames_ = list(malaya_speech.generator.frames(
                y_, 30, 16000, append_ending_trail=False))
            frames_webrtc = [(frames[no], vad(frame))
                            for no, frame in enumerate(frames_)]
            y_ = malaya_speech.combine.without_silent(frames_webrtc)
            zero = np.zeros(((1*sr+4000)-y_.shape[0]))
            signal = np.concatenate((y_, zero))
            all_wave.append(signal)
            all_label.append(label)
    print("Loaded " + str(np.array(all_wave).shape) + " files!")
    return np.array(all_wave), np.array(all_label)

load_files()