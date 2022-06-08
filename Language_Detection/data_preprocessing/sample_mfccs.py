import os
import numpy as np
import pandas as pd
import random
from datetime import datetime

#   Languages [(str)]
#       en - English
#       zh - Mandarin
#       hi - Hindi
#       es - Spanish
#       fr - French
#       ar - Arabic
#       bn - Bengali
#       ru - Russian
#       pt - Portugese
#       ur - Urdu
#       id - Indonesian - Insufficient Samples
#       de - German
#       ja - Japanese
langs=['en','zh','hi','es','fr','ar','bn','ru','pt','ur','de','ja'] #12 languages no indonesian

#   Clip Length [(int)] - 3, 5, 10 (seconds)
clip_length=3

#   Title (str) - If None will names based on number of languages and sample length
title=None

#   Data  Split Sizes [(int)] - train, dev, test
#       One for each clip length provided
#       If None will default to splits below:
#           3 - [65000,2000,2000]
#           5 - [32000,1200,1200]
#           10 - [8500,400,400]
split_size=None

#   Data path (str) - points to location of mfccs
data_path='../mfccs'

#   Output path (str) - place to write output
output_path='../output'

##################################################################################################

n_langs=len(langs)

if title == None:
    title=str(n_langs)+'langs_'+str(clip_length)+'s'

if split_size==None:
    if clip_length==3:
        split_size=[65000,2100,2100]
    elif clip_length==5:
        split_size=[32000,1180,1180]
    elif clip_length==10:
        split_size=[8500,400,400]
    else:
        print('Invalid clip length')

data_splits=['train','val','test']

data=[]
data_summary=[]
new_langs=[]

N_train=[]
N_dev=[]
N_test=[]

n_langs=len(langs)
out_path=output_path+'/'+title

def append_plus(big,little):
    if big is None:
        big=little
    else:
        big=np.concatenate((big,little),axis=0)
    return big

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


def main():
    
    backup_dir(out_path)
    
    length_path=data_path+'/'+str(clip_length)+'s'
    
    if os.path.isdir(length_path)!=True:
        print('\nTarget path does not exist:\n'+length_path) 
        
    else:
        
        n_split=[]
        
        for i_split, split in enumerate(data_splits):
            print('\n\n'+split)
            split_path=length_path+'/'+split
            
            if os.path.isdir(split_path)!=True:
                print('\nTarget path does not exist:\n'+split_path) 
                
            else:
                   
                X=None
                Y=None
                
                x_shapes=[]
                y_shapes=[]
                
                n_min=int(9e9)
                n_lang=[]
                
                for i_lang, lang in enumerate(langs):
                    print(' '+lang)
                    lang_path=split_path+'/'+lang+'.npy'                        
                    
                    if os.path.isfile(lang_path)!=True:
                        print('\nTarget file does not exist:\n'+lang_path)
                        n_lang.append(0)
                        #print('\nLanguage '+lang+' not available for '+str(length)+'s '+split)
                    else:
                        data_lang=np.load(lang_path)
                        n_data=len(data_lang)
                        if len(data_lang) < split_size[i_split]:
                            print('Dataset is smaller than length requested!')
                        else:
                            random.shuffle(data_lang)
                            
                            x=data_lang[0:split_size[i_split]]
                            y=i_lang*np.ones(len(x)).astype(int)
                            
                            x_shapes.append(x.shape)                            
                            y_shapes.append(y.shape)
                            
                            print('  x: '+str(x.shape))
                            print('  y: '+str(y.shape))
                            
                            X=append_plus(X,x)
                            Y=append_plus(Y,y)

                        n_min=min(n_min,len(data_lang))   
            
            
            
            print('\n X: '+str(X.shape))
            print(' Y: '+str(Y.shape))
            
            n_split.append(n_min)
            np.save(out_path+'/X_'+split+'.npy',X)            
            np.save(out_path+'/Y_'+split+'.npy',Y)          
            
            data_summary=pd.DataFrame(list(zip(langs,x_shapes,y_shapes,range(n_langs))),
                                      columns=('Language','X Shape','Y Shape','Y Index'))
            print(data_summary)
            data_summary.to_csv(out_path+'/'+split+'_summary.csv',index=False)
        np.save(out_path+'/Y_key.npy',langs)                             
    
if __name__ == '__main__':
    main()
