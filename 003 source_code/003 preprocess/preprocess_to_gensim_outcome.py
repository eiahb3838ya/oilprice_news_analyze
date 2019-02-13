# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:27:33 2019

@author: eiahb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gensim,os,sklearn,tqdm

from collections import Counter
from datetime import datetime


from gensim import corpora , models

# 1.load data
#load all token txt
txt_dir="D:/work/fortune_street/002 news_analyze/002 data/002 corpus_data/token_txt_data/"
list_txt_source=list(os.walk(txt_dir))[0][1]
print(list_txt_source)

#load stop words where
temp_stopwords=[]

def read_news_txt_file(aFilepath):
    token_list=[]
    with open(aFilepath,"r",encoding='utf8') as f:
        news_title=f.readline()
        line = news_title
        while line :
            line = f.readline()
            token_list.append(line.strip())
    return(news_title,token_list)
    
def clean_stop_words(aListOfTokens):
    new_l=[token for token in aListOfTokens if token not in temp_stopwords]
    return(new_l)
    
full_list=[]
for aSource in list_txt_source:
    for subpath , null , list_news_filenames in tqdm.tqdm(list(os.walk(txt_dir+aSource))[1:]):
        news_publish_date=subpath.split("\\")[1]
        source=subpath.split("\\")[0].split("/")[-1]
        for aFilename in list_news_filenames:
            aFilepath="/".join([subpath, aFilename])
            news_title,news_tokens_list=(read_news_txt_file(aFilepath))
            
            out_dict={'news_publish_date':news_publish_date,
                      'news_title':news_title,
                      'news_tokens_list':news_tokens_list,
                      'source':source}
            full_list.append(out_dict)
full_news_df=pd.DataFrame(full_list)
full_news_df['news_tokens_list_stripped']=full_news_df.news_tokens_list.apply(clean_stop_words)

# 2.gensim
## build dictionary

DICT_FOLDER = "D:/work/fortune_street/002 news_analyze/002 data/003 outcome_data/gemsim_outcome/dictionary/"
CORPUS_FOLDER="D:/work/fortune_street/002 news_analyze/002 data/003 outcome_data/gemsim_outcome/corpus/"
saving_date=datetime.today().date()
print('Folder "{}" will be used to save dictionary and corpus for {}'.format(DICT_FOLDER,saving_date))
if not os.path.exists(CORPUS_FOLDER):
    os.makedirs(CORPUS_FOLDER)
if not os.path.exists(DICT_FOLDER):
    os.makedirs(DICT_FOLDER)
    
#save dictionary
dictionary = corpora.Dictionary(full_news_df.news_tokens_list_stripped)
dictionary.save(os.path.join(DICT_FOLDER, str(saving_date)+'.dict'))  # store the dictionary, for future reference
print("dictionary saved at:"+os.path.join(DICT_FOLDER, str(saving_date)+'.dict'))
## transform token_list to corpus

corpus = [dictionary.doc2bow(text) for text in full_news_df.news_tokens_list]
corpora.MmCorpus.serialize(CORPUS_FOLDER+str(saving_date)+'.mm', corpus)
print("corpus saved at:"+CORPUS_FOLDER+str(saving_date)+'.mm')