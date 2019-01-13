#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import re
import pandas as pd
import numpy as np
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pprint import pprint
from collections import Counter
from math import *
from datetime import datetime
from common_class import *
from common_class import Preprocessor,word_to_vector,get_content,MyStopWord
from progressbar import progressbar
# import matplotlib.pyplot as plt
import sys


# In[3]:
window_size = 5
TRAIN_START_DATE = "2017-12"

TRAIN_INTIVAL = 8
TEST_INTIVAL = 2

TF_TARGET_THRESHOLD=2
OTHER_TARGET_THRESHOLD=25
DATA_ROOT = "../../../data/"
most_common_count = 800

effective_date_path = DATA_ROOT+"crude_oil_price/effective_news_date_days_before_and_after.csv"
from_date = TRAIN_START_DATE
to_date = str(np.datetime64(TRAIN_START_DATE) +
              np.timedelta64(TRAIN_INTIVAL-1, 'M'))

test_from_date = str(np.datetime64(TRAIN_START_DATE) +
              np.timedelta64(TRAIN_INTIVAL, 'M'))
test_to_date = str(np.datetime64(test_from_date)+np.timedelta64(TEST_INTIVAL, 'M'))


print("########################\n"+"from_date", from_date, "\nto_date", to_date, "\ntest_from_date",
      test_from_date, "\ntest_to_date", test_to_date+"\n########################")


# In[4]:


raw_df = pd.read_csv(DATA_ROOT+"crawler_news_data/oilprice_news.csv")
raw_df_cnbc = pd.read_csv(DATA_ROOT+"crawler_news_data/cnbc_oil_news.csv")
data_df = raw_df.sort_values(
    by="publish_datetime", ascending=True).set_index('publish_datetime')
data_df_cnbc = raw_df_cnbc.sort_values(
    by="story_publish_datetime", ascending=True).set_index('story_publish_datetime')
data_df_oilprice = pd.DataFrame(
    {"date": raw_df.publish_datetime, "content": raw_df.content})
data_df_cnbc = pd.DataFrame(
    {"date": raw_df_cnbc.story_publish_datetime, "content": raw_df_cnbc.story_full_article})
data_df_oilprice_cnbc = data_df_oilprice.append(data_df_cnbc)
data_df_oilprice_cnbc = data_df_oilprice_cnbc.sort_values(
    by="date", ascending=True).set_index('date')
raw_content = get_content(data_df_oilprice_cnbc)


# In[5]:


train_content = raw_content[from_date:to_date]


# ## 建立stopword

# In[6]:


all_year_preprocessor = Preprocessor(content=train_content)
all_year_preprocessor.to_counter()
mystopword = MyStopWord(
    content_counter=all_year_preprocessor.counter, most_common=87)
mystopword.define('c')
mystopword.keep('demand')


# ## 用Target corpus - Other corpus find dictionary

# ### 1. effectivate news date

# In[7]:


effective_news_df = pd.read_csv(effective_date_path)
effective_news_date = effective_news_df['date']
effective_news_date = pd.DatetimeIndex(effective_news_date)
effective_news_date


# ### 2. find target and other corpus

# In[9]:


train_content = raw_content[from_date:to_date]
target_content = train_content.loc[train_content.index.isin(
    effective_news_date.values)]

other_content = train_content.loc[~train_content.index.isin(
    effective_news_date.values)]
target_preprocessor = Preprocessor(content=target_content)
other_preprocessor = Preprocessor(content=other_content)
target_preprocessor.to_counter()
other_preprocessor.to_counter()


# ### 3. find bigram dictionary

# In[10]:


window_size = 5
target_bigram = Bigram(token=target_preprocessor.tokens)
other_bigram = Bigram(token=other_preprocessor.tokens)
target_bigram.count_word_pair_with_windows(
    mystopword=mystopword, window_size=window_size)
other_bigram.count_word_pair_with_windows(
    mystopword=mystopword, window_size=window_size)
target_corpus_set=set([key for key,times in target_bigram.pair_counts.most_common(most_common_count) if times>TF_TARGET_THRESHOLD])
other_corpus_set=set([key for key,times in other_bigram.pair_counts.items() if times>OTHER_TARGET_THRESHOLD])

pairwise_dictionary = target_corpus_set - other_corpus_set
print("len(pairwise_dictionary)",len(pairwise_dictionary))

# ## 4. word to vector
# In[12]:


train_preprocessor = Preprocessor(content=train_content)
train_preprocessor.to_counter()
train_vs = word_to_vector(train_preprocessor.tokens,
                          pairwise_dictionary, mystopword, window_size)
train_vs_df = pd.DataFrame(train_vs)
train_vs_df = train_vs_df.set_index(pd.DatetimeIndex(train_content.index))


# In[13]:

print("there are ", len(target_content), " out of ", len(
    train_content), "content in the effective days in training set")
print("train target vs sum:", train_vs_df.loc[train_content.index.isin(
    effective_news_date.values)].sum(axis=1).mean())
print("train other vs sum:", train_vs_df.loc[~train_content.index.isin(
    effective_news_date.values)].sum(axis=1).mean())


# In[14]:


train_vs_df['tags'] = 0
train_vs_df.loc[train_vs_df.index.isin(effective_news_date.values), 'tags'] = 1
train_vs_df.to_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_train.csv")


# ## Generate Test Data

# In[17]:


if len(test_to_date):
    test_content = raw_content[test_from_date:test_to_date]
else:
    test_content = raw_content[test_from_date:]
# effective_date = pd.read_csv(
#     "../../data/crude_oil_price/effective_news_date_from_2013.csv")


# In[18]:


test_preprocessor = Preprocessor(content=test_content)
test_preprocessor.to_counter()
test_vs = word_to_vector(test_preprocessor.tokens,
                         pairwise_dictionary, mystopword, window_size)
test_vs_df = pd.DataFrame(test_vs)
test_vs_df = test_vs_df.set_index(pd.DatetimeIndex(test_content.index))
test_vs_df['tags'] = 0
###############挫賽了拉
test_vs_df.loc[test_vs_df.index.isin(effective_news_date.values),'tags'] = 1
###############挫賽了拉
test_vs_df.to_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_test.csv")


# In[19]:

print("there are ", len(test_vs_df.loc[test_vs_df.index.isin(effective_news_date.values)]), " out of ", len(
    test_content), "content in the effective days in testing set")
print("test target vs sum:", test_vs_df.loc[test_content.index.isin(
    effective_news_date.values)].sum(axis=1).mean())
print("test other vs sum:", test_vs_df.loc[~test_content.index.isin(
    effective_news_date.values)].sum(axis=1).mean())


import bigram_predict
