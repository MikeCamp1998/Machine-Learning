# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:11:11 2022

@author: nagra
"""

#The duration is equal to the number of frames divided by the framerate (frames per second):
import os
import wave
import contextlib
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import warnings

warnings.filterwarnings("ignore")

target_length=10
target_lengths=[3,5,10]
overlap=0.5

data_dir='C:/bin/voxlingua107/'

for lang in os.listdir(data_dir):
    lang_dir=data_dir+lang
    #if os.path.isdir(lang_dir)==True:
    if lang=='en':
        print('\n'+lang)
        print(lang_dir)
        
        mfccs=[]
        mels=[]
        
        for i in range((len(target_lengths))):
            mfccs.append([])
            mels.append([])
        
        i_file=0
        
        for f in os.listdir(lang_dir):
            
            # if i_file>=100:
            #     for i_length, target_length in enumerate(target_lengths):
            #         mfccs_i=np.array(mfccs[i_length])
            #         mels_i=np.array(mels[i_length])
            #         np.save(lang+'_'+str(target_length)+'s_mfccs.npy', mfccs_i)
            #         np.save(lang+'_'+str(target_length)+'s_mels.npy', mels_i)
            #     break
            
            if os.path.splitext(f)[-1]=='.wav':
                file_path=lang_dir+'/'+f
                
                sound, rate = librosa.load(file_path)
                length = librosa.get_duration(sound)
                
                for i_length, target_length in enumerate(target_lengths):
                
                    if length >= target_length:
                        
                        i_file += 1
                        
                        if i_file % 1000 == 0:
                            print(' file: '+str(i_file))
                            for i_length, target_length in enumerate(target_lengths):
                                    mfccs_i=np.array(mfccs[i_length])
                                    mels_i=np.array(mels[i_length])
                                    np.save(lang+'_'+str(target_length)+'s_mfccs_'+i_file+'.npy', mfccs_i)
                                    np.save(lang+'_'+str(target_length)+'s_mels_'+i_file+'.npy', mels_i)
                        
                        # time = np.array(range(0,len(sound)))/rate
                        # fig, ax = plt.subplots(1,1)
                        # ax.plot(time,sound)
                        # fig.savefig(lang+'_full.png')
                        # plt.close(fig)
                        
                        if i_file>17000:
                        
                            i_clip = 0
                            clip_start = int(0)
                            clip_end = int(clip_start+target_length*rate)
                            
                            while clip_end <= len(sound):
                                
                                clip = sound[clip_start:clip_end]
                                time = np.array(range(clip_start,clip_end))/rate
                                
                                sgram = librosa.stft(clip)
                                sgram_mag, _ = librosa.magphase(sgram)
                                mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=rate)
                                mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
                                mfcc = librosa.feature.mfcc(clip, sr=rate)
                                mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
                                
                                mfccs[i_length].append(mfcc)
                                mels[i_length].append(mel_sgram)
                                
                                # fig, ax = plt.subplots(1,1)
                                # ax.plot(time,clip)
                                # fig.savefig(lang+'_'+str(i_clip)+'.png')
                                # plt.close(fig)
                                
                                i_clip += 1
                                clip_start += int(overlap * target_length * rate)
                                clip_end = int(clip_start+target_length*rate)
                
        for i_length, target_length in enumerate(target_lengths):
            mfccs_i=np.array(mfccs[i_length])
            mels_i=np.array(mels[i_length])
            np.save(lang+'_'+str(target_length)+'s_mfccs.npy', mfccs_i)
            np.save(lang+'_'+str(target_length)+'s_mels.npy', mels_i)