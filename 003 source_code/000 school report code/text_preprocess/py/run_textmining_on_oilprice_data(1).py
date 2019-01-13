
# coding: utf-8

# In[58]:


import nltk,re
import pandas as pd
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pprint import pprint
from collections import Counter
from math import *
from datetime import datetime
# nltk.download('punkt')


# In[59]:


def get_content(data_df):
    content=data_df.content
    content.index = pd.DatetimeIndex(content.index)
    content=content.dropna(how="any")
    return(content)
def stem_and_other_stuff(each_news):
    ps=PorterStemmer()
    return([ps.stem(word.lower()) for word in each_news if word.isalpha()])
def check_alpha_tolower(each_news):
    return([word.lower() for word in each_news if word.isalpha()])

def to_counter(this_year_content,stem=False):
    token_content=pd.Series()
    token_content=this_year_content.apply(word_tokenize)
    ps=PorterStemmer()
    if stem:        
        stemmed_content=token_content.apply(stem_and_other_stuff)
    else:
        stemmed_content=token_content.apply(check_alpha_tolower)
    content_counter = Counter()
    for news in stemmed_content:
        content_counter.update(news)
    return(stemmed_content,content_counter)


class MyStopWord:
    def __init__(self,content_counter,most_common=100,stop_word=None):
        from nltk.corpus import stopwords
        self.counter_stop_word=[word for word,time in content_counter.most_common(most_common)]
        self.user_keep=[]
        self.user_define=[]
        if stop_word:
            self.stop_word=stop_word
        else:
            self.stop_word=set(self.counter_stop_word+stopwords.words('english')) 
    def keep(self,word):
        self.user_keep.append(word)
        self.stop_word.discard(word)
    def define(self,word):
        self.user_define.append(word)
        self.stop_word.add(word)
        


# In[61]:


raw_df = pd.read_csv("../../data/crawler_news_data/oilprice_news.csv")
data_df=raw_df.sort_values(by="publish_datetime",ascending=True).set_index('publish_datetime')
content = get_content(data_df)
effective_news_df=pd.read_csv("../../data/crude_oil_price/effective_news_date.csv")
effective_news_date = effective_news_df['date']
effective_news_date=pd.DatetimeIndex(effective_news_date)


# # To_Counter

# In[63]:


this_year_content=content["2017-11":"2018-09"]
this_year_token, this_year_counter = to_counter(this_year_content)


# # get token and target token

# In[64]:


target_content = content.loc[effective_news_date.values].dropna(how="any")
other_content=content.loc[~this_year_content.index.isin(effective_news_date.values)].dropna(how="any")
target_token,target_counter=to_counter(target_content)
other_token,other_counter=to_counter(other_content)
content_token,content_counter=to_counter(content)


# # define stop words

# In[65]:


mystopword=MyStopWord(content_counter=content_counter,most_common=87)
mystopword.define('c')
mystopword.keep('demand')


# # get corpus set

# In[66]:


TF_OTHER_THRESHOLD=20
TF_TARGET_THRESHOLD=5

other_corpus_set=set(key for key,times in other_counter.items() if times>TF_OTHER_THRESHOLD)-mystopword.stop_word
target_corpus_set=set(key for key,times in target_counter.items() if times>TF_TARGET_THRESHOLD)-mystopword.stop_word
target_corpus_set-other_corpus_set


# # pairwise experiment

# In[68]:


def count_word_pair_with_windows(target_token,window_size):
    stop_word = mystopword.stop_word
    target_pair_counts = Counter()
    target_pair_distance_counts = Counter()
    for tokens in target_token:
        for i in range(len(tokens) - 1):
            for distance in range(1, window_size):
                if i + distance < len(tokens):
                    w1 = tokens[i]
                    w2 = tokens[i + distance]
                    if w1 not in stop_word and w2 not in stop_word:
                        target_pair_distance_counts[(w1, w2, distance)] += 1
                        target_pair_counts[(w1, w2)] += 1
    return(target_pair_distance_counts,target_pair_counts)


# In[70]:


window_size = 5
target_pair_distance_counts,target_pair_counts = count_word_pair_with_windows(target_token,window_size)
other_pair_distance_counts,other_pair_counts = count_word_pair_with_windows(other_token,window_size)
pairwise_with_windows = set(target_pair_counts.most_common(100)) - set(other_pair_counts)
pairwise_with_windows_list=[pair for pair,count in pairwise_with_windows]


# In[72]:


pairwise_df = pd.DataFrame(pairwise_with_windows_list)
pairwise_df.to_csv("../../data/wordpair_result/bigram_result_this_year_oilprice.csv",index=False)


# ## Word to Vector

# In[73]:


# pairwise_chose_df = pd.read_excel("../../data/wordpair_result/bigram_result_excel.xlsx")
# pairwise_chosen_tuple = [tuple(x[:2]) for x in pairwise_chose_df.values]


# In[74]:


def word_to_vector(this_year_token,pairwise_with_windows_list,mystopword):
    this_year_vs=[]
    # pairwise_with_windows_list = pairwise_chosen_tuple
    for tokenized_article in this_year_token:
        finder = nltk.BigramCollocationFinder.from_words([word for word in tokenized_article if word not in mystopword.stop_word],window_size=window_size)
        this_vs= {key: 0 for key in pairwise_with_windows_list}
        for pair,times in finder.ngram_fd.items():
            if pair in this_vs.keys():
                this_vs[pair]=times
        this_year_vs.append(this_vs)            
    return(this_year_vs)


# In[75]:


this_year_vs = word_to_vector(this_year_token,pairwise_with_windows_list,mystopword)
this_year_vs_df=pd.DataFrame(this_year_vs)
this_year_vs_df=this_year_vs_df.set_index(pd.DatetimeIndex(this_year_content.index))

print("target:",this_year_vs_df.loc[effective_news_date.values].sum(axis=1).mean())
print("other:",this_year_vs_df.loc[~this_year_content.index.isin(effective_news_date.values)].sum(axis=1).mean())


# In[76]:


filename = "../../data/wordpair_result/oilprice_pairwise_df_window_"+str(window_size)+".csv"
oilprice_pairwise_df = pd.DataFrame({"pairwise": pairwise_with_windows_list})
oilprice_pairwise_df['pairwise1']=[x for x,y in oilprice_pairwise_df.pairwise]
oilprice_pairwise_df['pairwise2']=[y for x,y in oilprice_pairwise_df.pairwise]
oilprice_pairwise_df.to_csv(filename,sep=";",index = False)


# In[21]:


def content_to_pairwise_vector(test_content,window_size,pairwise_with_windows_list):
    test_token,test_counter = to_counter(test_content,False)
    test_pairwise_with_distance,test_pairwise = count_word_pair_with_windows(test_token,window_size)
    test_vs = word_to_vector(test_token,pairwise_with_windows_list,mystopword)
    test_vs_df = pd.DataFrame(test_vs,index = test_content.index)
    return(test_vs_df)


# In[77]:


train_filename = "../../data/train_test_dataset/this_year_oilprice_window_"+str(window_size)+"train.csv"
this_year_vs_df['tags'] = 0
this_year_vs_df.loc[this_year_vs_df.index.isin(effective_news_date.values),'tags']=1
this_year_vs_df.to_csv(train_filename,index = False)


# ## Use Effective date as test data

# In[28]:


test_filename = "../../data/train_test_dataset/this_month_oilprice_window_"+str(window_size)+"test.csv"
effective_date = pd.read_csv("effective_news_date_from_2013.csv")
test_target_date = effective_date.loc[effective_date.date>"2018-10-01","date"]
test_content = content["2018-10-01":]
test_vs_df = content_to_pairwise_vector(test_content,window_size,pairwise_with_windows_list)
test_vs_df['tags'] = 0
test_vs_df.loc[test_vs_df.index.isin(test_target_date)] =1
test_vs_df.to_csv(test_filename,index=False)

