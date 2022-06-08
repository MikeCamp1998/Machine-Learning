import os
import gc
import time
import pandas as pd
import numpy as np
import soundfile as sf
import scipy.signal as signal
import sklearn
import IPython.display as ipd
import keras
import keras.backend as K
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow_io as tfio

from keras import layers
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from matplotlib.pyplot import imshow
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle

# All useful functions stored here

#------------Dataset Loading functions--------------#

def get_label(filepath):
    label = filepath.split('/')[-1][:2]
    id = -1
    if label == "en":
      id = 0
    elif label == "es":
      id = 1
    elif label == "de":
      id = 2
    return id

def get_mel_spectrogram(filepath):
    samples, sample_rate = librosa.load(filepath, sr=None)                         # Load the audio file with librosa
    sgram = librosa.stft(samples)                                                  # Compute the short term fourier transform to create a spectrogram
    sgram_mag, _ = librosa.magphase(sgram)                                         # Get the spectrogram's magnitude
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)  # Convert to a Mel Scale
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)               # Get the mel spectrogram with amplitude in the dB scale for better viewing
    return (mel_sgram)

def get_mfcc(filepath):
    samples, sample_rate = librosa.load(filepath, sr=None) 
    mfcc = librosa.feature.mfcc(samples, sr=sample_rate)                            # Get the Mel Frequency Cepstral Coefficients (MFCC)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)                                # Center MFCC coefficient dimensions to the mean and unit variance
    #mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    #np.subtract(mfcc,np.mean(mfcc))
    return mfcc

def get_mel_and_label(filepath):
    label = get_label(filepath)
    mel = get_mel_spectrogram(filepath)
    return mel, label

def get_mfcc_and_label(filepath):
    label = get_label(filepath)
    mfcc = get_mfcc(filepath)
    return mfcc, label

def load_mel_features(df):
    X = []
    y = []
    for filepath in df['Filepath']:
        mel, label = get_mel_and_label(filepath)
        X.append(mel)
        y.append(label)
    feature_df = pd.DataFrame({"mel": X, "Class ID": y})
    return feature_df

def load_mfcc_features(df):
    X = []
    y = []
    for filepath in df['Filepath']:
        mfcc, label = get_mfcc_and_label(filepath)
        X.append(mfcc)
        y.append(label)
    feature_df = pd.DataFrame({"mfcc": X, "Class ID": y})
    return feature_df

def load_y_test(df):
    y_test = []

    for filepath in df['Filepath']:
        label = get_label(filepath)
        y_test.append(label)

    y_test = np.array(y_test)
    return y_test

#---------------------------------------------------#

#--------------Graphing functions-------------------#

def get_time_graph(filepath):
    samples, sample_rate = librosa.load(filepath, sr=None)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(samples, sr=sample_rate)
    plt.title('Time Domain Data')
    plt.ylabel('Amplitude')
    plt.xlabel('Time in seconds')

def get_mel_spec_graph(filepath):
    samples, sample_rate = librosa.load(filepath, sr=None)
    mel_sgram = get_mel_spectrogram(filepath)
    librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')

def get_mfcc_graph(filepath):
    samples, sample_rate = librosa.load(filepath, sr=None)
    mfcc = get_mfcc(filepath)
    librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')

#----------------------------------------------------#

#------------batch preprocessing functions-----------#

def get_freq_and_time_mask(mfcc, freq_max_range, time_max_range):
    freq_mask = tfio.audio.freq_mask(mfcc, param=freq_max_range)
    freq_and_time_mask = tfio.audio.time_mask(freq_mask, param=time_max_range)
    return freq_and_time_mask

def batch_generator(X, y, batch_size, num_steps):
    idx=1
    while True: 
        yield load_batch(X, y, idx-1, batch_size) ## Yields data
        if idx < num_steps:
            idx+=1
        else:
            idx=1

def load_batch(X, y, idx, batch_size):
    start_idx = idx * batch_size
    end_idx = start_idx + batch_size
    X_batch = X[start_idx : end_idx]
    y_batch = y[start_idx : end_idx]
    X_batch, y_batch = shuffle(X_batch, y_batch)
    return X_batch, y_batch


def batch_generator_2(X, y, batch_size, num_steps):
    idx=1
    while True: 
        yield load_batch_2(X, y, idx-1, batch_size) ## Yields data
        if idx < num_steps:
            idx+=1
        else:
            idx=1

def load_batch_2(X, y, idx, batch_size):
    start_idx = idx * batch_size
    end_idx = start_idx + batch_size
    X_batch = X[start_idx : end_idx]
    y_batch = y[start_idx : end_idx]

    X_batch_aug = []
    y_batch_aug = []

    for i in range(X_batch.shape[0]):  
        for j in range(1):
            mfcc_aug = get_freq_and_time_mask(X_batch[i], 50, 12)
            X_batch_aug.append(mfcc_aug)
            y_batch_aug.append(y_batch[i])

    X_batch_aug = np.array(X_batch_aug)
    y_batch_aug = np.array(y_batch_aug)

    X_batch = np.concatenate((X_batch, X_batch_aug))
    y_batch = np.concatenate((y_batch, y_batch_aug))

    X_batch, y_batch = shuffle(X_batch, y_batch)
    return X_batch, y_batch

#----------------------------------------------------#

# def mel_generator(X, y, batch_size):
#     features = []
#     labels = []
#     batch_count = 0
#     while True: 
#         for idx in range(X.shape[0]):
#             features.append(X[idx])
#             labels.append(y[idx])
#             batch_count += 1
#             if batch_count > batch_size:
#                 X_batch = np.array(features, dtype='float32')
#                 y_batch = np.array(labels, dtype='float32')
#                 yield (X_batch, y_batch)
#                 features = []
#                 labels = []
#                 batch_count = 0