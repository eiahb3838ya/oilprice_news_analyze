# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:02:39 2019

@author: Evan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os,sklearn,tqdm,nltk

from collections import Counter
from datetime import datetime

from gensim import corpora
from gensim import models
from scipy.sparse import csr_matrix

import gensim
nltk.download('averaged_perceptron_tagger')

## 1.set args
ROOT_DIR_OF_PROJECT="C:\\Users\\Evan\\MyFile\\Fortune-street\\007 oil_price\\oilprice_news_analyze\\"


# 'CC'表示并列连词，'RB'表示副词，'IN'表示介词，'NN'表示名词，'JJ'表示形容词
def pos_filter(aListToken):
    remove_pos_list=['IN','DT','FW','CC','TO','MD','WP','LS','EX','PDT','RP','PRP$']
    text1=aListToken
    if '' in text1:
        text1.remove('')
    pos_list=nltk.pos_tag(text1)
    return([word for word,pos in pos_list if pos not in remove_pos_list])
    
    
    
    
def read_news_txt_file(aFilepath):
    token_list=[]
    with open(aFilepath,"r",encoding='utf8') as f:
        news_title=f.readline()
        line = news_title
        while line :
            line = f.readline()
            token_list.append(line.strip())
    return(news_title,token_list)


counter_dir=ROOT_DIR_OF_PROJECT+"002 data/003 outcome_data/word_counter_data/"
txt_dir=ROOT_DIR_OF_PROJECT+"002 data/002 corpus_data/token_txt_data/"
# C:\Users\Evan\MyFile\Fortune-street\007 oil_price\oilprice_news_analyze\002 data\002 corpus_data\token_txt_data

list_counter_filenames=list(os.walk(counter_dir))[0][2]
print(list_counter_filenames)
list_txt_source=list(os.walk(txt_dir))[0][1]
print(list_txt_source)

full_list=[]
for aSource in list_txt_source:
    for subpath , null , list_news_filenames in tqdm.tqdm(list(os.walk(txt_dir+aSource))[1:]):
        news_publish_date=subpath.split("\\")[1]
        source=subpath.split("\\")[0].split("/")[-1]
        for aFilename in list_news_filenames:
            aFilepath="/".join([subpath, aFilename])
            news_title,news_tokens_list=(read_news_txt_file(aFilepath))
            
            ##apply pos filter and save
            pos_filtered_Filepath=aFilepath.replace('token_txt_data','token_pos_filtered_txt_data')
            pos_filtered_dir='/'.join(pos_filtered_Filepath.split('/')[:-1])
            if not os.path.exists(pos_filtered_dir):
                os.makedirs(pos_filtered_dir)
            filtered=pos_filter(news_tokens_list)
            a_title,a_tokens=news_title,filtered
            if not os.path.exists(pos_filtered_Filepath):
                print("write")
            #pos filtered data already exist
                with open(pos_filtered_Filepath, "w",encoding="utf8") as text_file:
                    text_file.writelines(a_title.lower()+"\n")
                    for token in a_tokens:
                        text_file.write(token+"\n")
                    
            ##append to full list
            out_dict={'news_publish_date':news_publish_date,
                      'news_title':news_title,
                      'news_tokens_list':news_tokens_list,
                      'source':source,
                      'news_pos_filtered_tokens_list':filtered}
            
            full_list.append(out_dict)
            
full_news_df=pd.DataFrame(full_list)
print(full_news_df.head(10))