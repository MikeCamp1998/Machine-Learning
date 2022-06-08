# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:11:11 2022

@author: nagra
"""

#The duration is equal to the number of frames divided by the framerate (frames per second):
import sys
import os
import wave
import contextlib
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import warnings
import time
import sklearn
import random
from datetime import datetime

def backup_dir(dir_path):
    basedir = dir_path
    now = datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H%M%S')
    budir = basedir+'_'+timestamp
    
    if os.path.isdir(basedir):
        print("\ndirectory: "+basedir+" already exists...")
        os.rename(basedir,budir)
        print("...moved to "+budir)
        
    os.mkdir(basedir)
    print("\ndirectory: "+basedir+" created")

warnings.filterwarnings("ignore")

data_dir='D:\\voxlinga\\voxlingua107\\' #'C:/bin/voxlingua107/'
out_path='C:/bin/voxlingua107/data_split'
langs=['en','de','es','fr','zh','hi','ar','ja','pt','ru','vi','bn','ur','id']

target_lengths=[3,5,10]
dev_size=[700,600,400]
min_train=[65000,32000,8500]
#dev_size=[1,1,1]
overlap=0.5 #for train set only

n_bu=1000
n_status=1000

random.seed(2022)

data_splits=['val','test','train']

#dir(out_path)


sum_path=out_path+'/summary.csv'                     
print('Writing summary to: '+sum_path)

header='lang'

for split in data_splits: 
    for target_length in target_lengths:
        header=header+','+split+str(target_length)

#with open(sum_path,'a') as sum_file:
    #print(header,file=sum_file) 

n_files=[]
already_ran=[]

new_langs=[]
for lang in langs:
    lang_dir=data_dir+lang
    log_path=out_path+'/'+lang+'.log'
    if os.path.isdir(lang_dir)==True:
        file_list=os.listdir(lang_dir)
        n_files.append(len(file_list))
    if os.path.isfile(log_path)!=True:
        new_langs.append(lang)

langs=new_langs
        
     
print(langs)
print(n_files)

langs=[x for _, x in sorted(zip(n_files,langs))]
n_files=[x for _, x in sorted(zip(n_files,n_files))]
already_ran=[x for _, x in sorted(zip(n_files,already_ran))]



print(langs)
print(n_files)
sys.exit()   

if min_train==None:
    min_train=[]
    for i in range((len(target_lengths))):
        min_train.append(int(9e9))
    

# for length in target_lengths:
#     length_path=out_path+'/'+str(length)+'s'
#     os.mkdir(length_path)
#     for data_set in data_splits:
#         data_set_path=length_path+'/'+data_set
#         os.mkdir(data_set_path)       
        
t0 = time.time()

for i_lang, lang in enumerate(langs):
    
    lang_dir=data_dir+lang
    if os.path.isdir(lang_dir)!=True:
        print('\nTarget path does not exist or is not a directory:\n'+lang_dir)        
    else:            
        print('\n'+lang)
        log_path=out_path+'/'+lang+'.log'                      
        print('Writing log to: '+log_path)
        
        with open(log_path,'a') as log_file:
            print('Language log for '+lang,file=log_file)
        
        file_list=os.listdir(lang_dir)
        random.shuffle(file_list)
        
        n_samples=[0,0,0]
        i_split=0     
        
        mfccs=[]
        for i in range((len(target_lengths))):
            mfccs.append([])
            
        i_file=0
        
        summary=lang
        
        while i_split < 3 and i_file < len(file_list):
                                  
            if i_file % n_status == 0:   
                         
                with open(log_path,'a') as log_file:                        
                    now = datetime.now()
                    timestamp = now.strftime('%m/%d/%Y %H:%M:%S')                            
                    print('\n'+timestamp,file=log_file)
                    print(str(i_file)+' files processed',file=log_file)
                    for i_length, target_length in enumerate(target_lengths):
                        
                        if i_split < 2:
                            sample_target=dev_size[i_length]
                        else:
                            sample_target=min_train[i_length]
                            
                        mfccs_i=np.array(mfccs[i_length]) 
                        print(str(len(mfccs_i))+'/'+str(sample_target)+' samples of '+str(target_length)+' seconds',file=log_file)
            
            f = file_list[i_file]
        
            if os.path.splitext(f)[-1]=='.wav':
                file_path=lang_dir+'/'+f                             
                                                                       
                sound, rate = librosa.load(file_path)
                length = librosa.get_duration(sound)
                                
                for i_length, target_length in enumerate(target_lengths):
                    
                    i_clip = 0
                    clip_start = int(0)
                    clip_end = int(clip_start+target_length*rate)
                    
                    while clip_end <= len(sound):
                        
                        clip = sound[clip_start:clip_end]
                        
                        mfcc = librosa.feature.mfcc(clip, sr=rate)
                        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
                        
                        mfccs[i_length].append(mfcc)
                        
                        i_clip += 1
                        
                        if i_split < 2:
                            clip_start += int( target_length * rate)
                        else:
                            clip_start += int( (1-overlap) * target_length * rate)
                            
                        clip_end = int(clip_start+target_length*rate)
                       
                criteria=0
                                    
                for i_length, target_length in enumerate(target_lengths):
                    if i_split < 2:
                        if len(mfccs[i_length])>=dev_size[i_length]:
                            criteria+=1
                    else:
                        if len(mfccs[i_length])>=min_train[i_length]:
                            criteria+=1
                    
                if criteria >= len(target_lengths):
                    
                    with open(log_path,'a') as log_file:                        
                        now = datetime.now()
                        timestamp = now.strftime('%m/%d/%Y %H:%M:%S')                            
                        print('\n'+timestamp,file=log_file)
                        print(data_splits[i_split]+' data set',file=log_file)
                    
                    for i_length, target_length in enumerate(target_lengths):
                            mfccs_i=np.array(mfccs[i_length])                                                        
                            mfcc_path=out_path+'/'+str(target_length)+'s/'+data_splits[i_split]+'/'+lang+'.npy'
                            np.save(mfcc_path, mfccs_i)
                            
                            with open(log_path,'a') as log_file:
                                print(str(len(mfccs_i))+'/'+str(dev_size[i_length])+' samples of '+str(target_length)+' seconds written to:\n'+mfcc_path,file=log_file)
                            
                            summary=summary+','+str(len(mfccs_i))
                            
                    i_split +=1   
                    
                    mfccs=[]
                    for i in range((len(target_lengths))):
                        mfccs.append([])                                  
                        
                i_file += 1
                
            if i_file >= len(file_list):
                
                with open(log_path,'a') as log_file:
                    now = datetime.now()
                    timestamp = now.strftime('%m/%d/%Y %H:%M:%S')                            
                    print('\n'+timestamp,file=log_file)
                    print(data_splits[i_split]+' data set:',file=log_file)
                    print('Expended dataset ('+str(i_file)+' files) before reaching former minimum training set for each length.',file=log_file)
                    
                    
                print('\nExpended dataset ('+str(i_file)+' files) before reaching former minimum training set for each length.')
                    
                for i_length, target_length in enumerate(target_lengths):
                    
                    mfccs_i=np.array(mfccs[i_length])                                                        
                    mfcc_path=out_path+'/'+str(target_length)+'s/'+data_splits[i_split]+'/'+lang+'.npy'
                    np.save(mfcc_path, mfccs_i)
                    
                    with open(log_path,'a') as log_file:
                        print(str(len(mfccs_i))+'/'+str(min_train[i_length])+' samples of '+str(target_length)+' seconds written to:\n'+mfcc_path,file=log_file)
                    
                    if len(mfccs[i_length]) < min_train[i_length]:
                        print(str(len(mfccs[i_length]))+' samples less than former minimum for '+str(target_length)+' seconds of '+str(min_train[i_length])+'.')
                        #min_train[i_length] = int( 1000 * np.ceil(1.1*len(mfccs[i_length])/1000) )
                        #print('New training set length set to '+str(min_train[i_length])+' samples.')
                        
                    summary=summary+','+str(len(mfccs_i))
                
                with open(sum_path,'a') as sum_file:
                    print(summary,file=sum_file)
                        
                    
                