
# coding: utf-8

# In[1]:


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


# In[5]:


class Preprocessor:
    def __init__(self,path="",content=False):
        if len(path)>1:
            raw_df = pd.read_csv(path)
            self.data_df = raw_df.sort_values(by="publish_datetime",ascending=True).set_index('publish_datetime')
            content=self.data_df.content
            content.index = pd.DatetimeIndex(content.index)
            content=content.dropna(how="any")
            self.content = content
        else:
            self.content = content
    def stem_and_other_stuff(self,each_news):
        ps=PorterStemmer()
        return([ps.stem(word.lower()) for word in each_news if word.isalpha()])
    def check_alpha_tolower(self,each_news):
        return([word.lower() for word in each_news if word.isalpha()])
    def get_content_from_date(self,from_date,to_date):
        self.content = self.content[from_date:to_date]
    def to_counter(self,stem=False):
        self.token_content=self.content.apply(word_tokenize)
        if stem:        
            self.tokens=self.token_content.apply(self.stem_and_other_stuff)
        else:
            self.tokens=self.token_content.apply(self.check_alpha_tolower)
        content_counter = Counter()
        for news in self.tokens:
            content_counter.update(news)
        self.counter = content_counter


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

class Unigram:
    def __init__(self,target_counter,other_counter):
        self.target_counter = target_counter
        self.other_counter = other_counter
        
    def get_different_corpus_set(self,mystopword,TF_OTHER_THRESHOLD=20,TF_TARGET_THRESHOLD=5):
        other_corpus_set=set(key for key,times in self.other_counter.items() if times>TF_OTHER_THRESHOLD)-mystopword.stop_word
        target_corpus_set=set(key for key,times in self.target_counter.items() if times>TF_TARGET_THRESHOLD)-mystopword.stop_word
        self.different_corpus_set = target_corpus_set-other_corpus_set

class Bigram:
    def __init__(self,token):
        self.token = token
    def count_word_pair_with_windows(self,window_size,mystopword):
        stop_word = mystopword.stop_word
        self.pair_counts = Counter()
        self.pair_distance_counts = Counter()
        for tokens in self.token:
            for i in range(len(tokens) - 1):
                for distance in range(1, window_size):
                    if i + distance < len(tokens):
                        w1 = tokens[i]
                        w2 = tokens[i + distance]
                        if w1 not in stop_word and w2 not in stop_word:
                            self.pair_distance_counts[(w1, w2, distance)] += 1
                            self.pair_counts[(w1, w2)] += 1

class BigramResult:
    def __init__(self,effective_news_date,tokens,window_size,target_bigram=False,other_bigram=False,pairwise_dictionary=""):
        if len(pairwise_dictionary)==0:
            self.pairwise_dictionary = set(target_bigram.pair_counts.most_common(100)) - set(other_bigram.pair_counts)
            self.pairwise_dictionary = [pair for pair,count in self.pairwise_dictionary]
        else:
            self.pairwise_dictionary = pairwise_dictionary
        self.window_size=window_size
        self.effective_news_date = effective_news_date
        self.tokens = tokens
    def dictionary_to_csv(self,filename):
        oilprice_pairwise_df = pd.DataFrame({"pairwise": self.pairwise_dictionary})
        oilprice_pairwise_df['pairwise1']=[x for x,y in oilprice_pairwise_df.pairwise]
        oilprice_pairwise_df['pairwise2']=[y for x,y in oilprice_pairwise_df.pairwise]
        oilprice_pairwise_df.to_csv(filename,sep=";",index = False)
    def word_to_vector(self,mystopword):
        self.this_year_vs=[]
        for tokenized_article in self.tokens:
            finder = nltk.BigramCollocationFinder.from_words([word for word in tokenized_article if word not in mystopword.stop_word],window_size=self.window_size)
            this_vs= {key: 0 for key in pairwise_with_windows_list}
            for pair,times in finder.ngram_fd.items():
                if pair in this_vs.keys():
                    this_vs[pair]=times
            self.this_year_vs.append(this_vs)
        self.this_year_vs_df= pd.DataFrame(self.this_year_vs)
        self.this_year_vs_df= self.this_year_vs_df.set_index(pd.DatetimeIndex(self.tokens.index))            
    def get_difference(self):
        print("target:",self.this_year_vs_df.loc[self.effective_news_date.values].sum(axis=1).mean())
        print("other:",self.this_year_vs_df.loc[~self.tokens.index.isin(self.effective_news_date.values)].sum(axis=1).mean())
    def to_train_csv(self,train_filename):
        self.this_year_vs_df['tags'] = 0
        self.this_year_vs_df.loc[self.this_year_vs_df.index.isin(self.effective_news_date.values),'tags']=1
        self.this_year_vs_df.to_csv(train_filename)


root = "../../../data/"

path = root + "crawler_news_data/oilprice_news.csv"
this_year_preprocessor = Preprocessor(path)

effective_news_df=pd.read_csv(root + "crude_oil_price/effective_news_date.csv")
effective_news_date = effective_news_df['date']
effective_news_date=pd.DatetimeIndex(effective_news_date)


## To_Counter
from_date = "2017-11"
to_date = "2018-09"
this_year_preprocessor.get_content_from_date(from_date,to_date)


## get token and target token

target_content = this_year_preprocessor.content.loc[effective_news_date.values].dropna(how="any")
other_content= this_year_preprocessor.content.loc[~this_year_preprocessor.content.index.isin(effective_news_date.values)].dropna(how="any")
target_preprocessor = Preprocessor(content=target_content)
other_preprocessor = Preprocessor(content=other_content)
target_preprocessor.to_counter()
other_preprocessor.to_counter()
this_year_preprocessor.to_counter()


## define stop words


mystopword=MyStopWord(content_counter=this_year_preprocessor.counter,most_common=87)
mystopword.define('c')
mystopword.keep('demand')


## try unigram

TF_OTHER_THRESHOLD=20
TF_TARGET_THRESHOLD=5

unigram = Unigram(target_preprocessor.counter,other_preprocessor.counter)
unigram.get_different_corpus_set(mystopword)


## pairwise experiment
window_size = 5
target_bigram = Bigram(target_preprocessor.tokens)
other_bigram = Bigram(other_preprocessor.tokens)
target_bigram.count_word_pair_with_windows(window_size,mystopword)
other_bigram.count_word_pair_with_windows(window_size,mystopword)


## Word to Vector
bigram_result = BigramResult(effective_news_date,this_year_preprocessor.tokens,window_size,target_bigram,other_bigram)
bigram_result.word_to_vector(mystopword)

filename = root + "wordpair_result/oilprice_pairwise_df_window_"+str(window_size)+".csv"
bigram_result.dictionary_to_csv(filename)
print("Done dictionary ", filename)

train_filename = root + "train_test_dataset/this_year_oilprice_window_"+str(window_size)+"_train.csv"
bigram_result.to_train_csv(train_filename)
print("Done train ",train_filename)

test_filename = root + "train_test_dataset/this_month_oilprice_window_"+str(window_size)+"_test.csv"
effective_date = pd.read_csv(root + "crude_oil_price/effective_news_date_from_2013.csv")
test_target_date = effective_date.loc[effective_date.date>"2018-10-01","date"]
path = root + "crawler_news_data/oilprice_news.csv"
year_preprocessor = Preprocessor(path)
year_preprocessor.get_content_from_date("2018-10","2018-11")

test_preprocessor = Preprocessor(content = year_preprocessor.content)
test_preprocessor.to_counter()
test_bigram_result = BigramResult(effective_news_date=test_target_date,tokens=test_preprocessor.tokens,window_size=window_size,pairwise_dictionary=bigram_result.pairwise_dictionary)
test_bigram_result.word_to_vector(mystopword=mystopword)
test_bigram_result.to_train_csv(test_filename)
print("Done test ", test_filename)

