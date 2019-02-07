# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:14:19 2019

@author: eiahb
"""
import os,pickle
import pandas as pd
import numpy as np
from collections import Counter
from nltk import word_tokenize
from nltk.stem import PorterStemmer



START_DATE="2018-01-01"
ROOT="D:/work/fortune_street/002 news_analyze/002 data/002 corpus_data/"

root_dir=ROOT
list_of_source=os.listdir(root_dir+"raw_csv_data/waiting_preprocess/")

class Preprocessor:
    def __init__(self,stem=False):
        self.stem=stem
        
    def stem_and_other_stuff(self,each_news,do_tokenize=False):
        if do_tokenize:
            each_news=word_tokenize(each_news)
        ps=PorterStemmer()
        return([ps.stem(word.lower()) for word in each_news if word.isalpha()])
        
    def check_alpha_tolower(self,each_news,do_tokenize=False):
        if do_tokenize:
            each_news=word_tokenize(each_news)
        return([word.lower() for word in each_news if word.isalpha()])
    
    def my_word_tokenize(self,content):
        try:
            return(word_tokenize(content))
        except:
            print("tokenize fail")
            return([])
    def get_tokens(self,content):
        token_content=content.apply(self.my_word_tokenize)
        if self.stem:        
            tokens=token_content.apply(self.stem_and_other_stuff)
        else:
            tokens=token_content.apply(self.check_alpha_tolower)
        return(tokens)
        
    def get_counter(self,tokens):
        content_counter = Counter()
        for news in tokens:
            content_counter.update(news)
        return(content_counter)
preprocessor=Preprocessor()
        
counter = Counter()
for aSource in list_of_source:
    list_of_filenames=os.listdir(root_dir+"raw_csv_data/"+aSource) 
    for aFile in list_of_filenames:
        news_data_df=pd.read_csv(root_dir+"raw_csv_data/"+aSource+"/"+aFile)
        news_contents=news_data_df['content']
        news_tokens=preprocessor.get_tokens(news_contents)
        news_counter=preprocessor.get_counter(news_tokens)
        news_titles=news_data_df['title']
        news_datetime=news_data_df['publish_datetime']
        
        counter.update(news_counter)
        
        savedir=root_dir+"token_txt_data/"+aSource+"/"+news_datetime[0]+"/"
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        for a_news_id,a_news in enumerate(zip(news_titles,news_tokens)):
            a_title,a_tokens=a_news
            with open(savedir+"news_"+str(a_news_id)+".txt", "w",encoding="utf8") as text_file:
                text_file.writelines(a_title.lower()+"\n")
                for token in a_tokens:
                    text_file.write(token+"\n")
                    
dictionary = counter
np.save(ROOT+'word_counter_data/word_counter.npy', dictionary) 

# Load
#counter = np.load('my_file.npy').item()
#     
