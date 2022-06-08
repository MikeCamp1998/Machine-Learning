import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
#       id - Indonesian - Inadequate samples produced
#       de - German
#       ja - Japanese
langs=['en','zh','hi','es','fr','ar','bn','ru','pt','ur','id','de','ja']

#   Clip Length [(int)] - 3, 5, 10 (seconds)
clip_lengths=[3,5,10]

data_path='../mfccs'
output_path='../output'

data_splits=['train','val','test']

def main():
    
    for clip_length in clip_lengths:
    
        length_path=data_path+'/'+str(clip_length)+'s'
        
        if os.path.isdir(length_path)!=True:
            print('\nTarget path does not exist:\n'+length_path) 
            
        else:
            
            n_split=[]
            n_mins=[]
            
            for i_split, split in enumerate(data_splits):
                split_path=length_path+'/'+split
                
                if os.path.isdir(split_path)!=True:
                    print('\nTarget path does not exist:\n'+split_path) 
                    
                else:
                    
                    n_min=9e9
                    n_lang=[]
                    
                    for i_lang, lang in enumerate(langs):
                        lang_path=split_path+'/'+lang+'.npy'                        
                        
                        if os.path.isfile(lang_path)!=True:
                            print('\nTarget file does not exist:\n'+lang_path)
                            n_lang.append(0)
                            #print('\nLanguage '+lang+' not available for '+str(length)+'s '+split)
                        else:
                            data_lang=np.load(lang_path)
                            n_lang.append(len(data_lang))
                            n_min=min(n_min,len(data_lang))
                            
                #n_split.append(np.array(n_lang))
                n_split.append(n_lang)
                n_mins.append(n_min)
                            
            data_summary=pd.DataFrame(list(zip(langs,n_split[0],n_split[1],n_split[2])),
                                      columns=('Language','n_train','n_dev','n_test'))   
            data_summary.to_csv(output_path+'/summary_'+str(clip_length)+'s.csv',index=False)
            print()
            print(data_summary)
            print(n_mins)
                        
if __name__ == '__main__':
    main()
