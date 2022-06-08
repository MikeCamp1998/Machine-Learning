from utils import *
import tensorflow_io as tfio
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

#Load Train Data
X_train = np.load('datasets/10s/X_train.npy')
y_train = np.load('datasets/10s/Y_train.npy')

def get_freq_and_time_mask(mfcc, freq_max_range, time_max_range):
    freq_mask = tfio.audio.freq_mask(mfcc, param=freq_max_range)
    freq_and_time_mask = tfio.audio.time_mask(freq_mask, param=time_max_range)
    return freq_and_time_mask

X_train_aug = []
y_train_aug = []

print("Converted to Lists")

for i in range(X_train.shape[0]):  #X_train.shape[0]
    for j in range(3):
        mfcc_aug = get_freq_and_time_mask(X_train[i], 100, 12)
        X_train_aug.append(mfcc_aug)
        y_train_aug.append(y_train[i])

print(X_train[0])
print(y_train[0])
print()
print(X_train_aug[102001])
print(y_train_aug[102001])
print()

X_train = np.array(X_train_aug)
y_train = np.array(y_train_aug)

print(X_train.shape)
print(y_train.shape)

np.save('datasets/10s/X_train_aug_4.npy', X_train)
np.save('datasets/10s/Y_train_aug_4.npy', y_train)

print("Preprocessing Complete")
