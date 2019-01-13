
# coding: utf-8

# In[3]:


import nltk,re
import pandas as pd
import numpy as np

from nltk import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from pprint import pprint
from collections import Counter
from math import *
from datetime import datetime
from progressbar import progressbar
# nltk.download('punkt')
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from gensim.summarization.textcleaner import get_sentences
from gensim.summarization.textcleaner import clean_text_by_sentences
from sklearn.feature_extraction.text import TfidfVectorizer



def get_content(data_df):
    content=data_df.content
    content.index = pd.DatetimeIndex(content.index)
    content=content.dropna(how="any")
    return(content)
class Preprocessor:
    def __init__(self,use_stem=True,use_summarize=True,summarize_word_count=200):
        self.use_stem=use_stem
        self.use_summarize=use_summarize
        self.summarize_word_count=summarize_word_count
    def stem_and_other_stuff(self,each_news):
        ps=PorterStemmer()
        return([ps.stem(word.lower()) for word in each_news if word.isalpha()])
    
    def check_alpha_tolower(self,each_news):
        return([word.lower() for word in each_news if word.isalpha()])
        
    def get_tokenized_content(self,content):
        return(content.apply(word_tokenize))
    
    def get_counter(self,content):
        tokenized_content_s=self.get_tokenized_content(content)
        content_counter = Counter()
        if self.use_stem:        
            stemmed_token_s=tokenized_content_s.apply(self.stem_and_other_stuff)
            for aStemmed_token in stemmed_token_s:
                content_counter.update(aStemmed_token)
#             self.counter = content_counter
        else:
            token_s=tokenized_content_s.apply(self.check_alpha_tolower)
            for aStemmed_token in token_s:
                content_counter.update(aStemmed_token)
        return(content_counter)
    
    def get_summarize(self,content,summarize_ratio=None):
        if summarize_ratio:
            return(content.apply(lambda txt:summarize(txt,word_count = summarize_word_count)))
        else:
            return(content.apply(lambda txt:summarize(txt,word_count = self.summarize_word_count)))
#             return(content.apply(lambda txt:summarize(txt,ratio = self.summarize_ratio)))
    def preprocess(self,content):
        if self.use_summarize:
            content=content.loc[content.apply(clean_text_by_sentences).apply(list).apply(len).apply(lambda x:x>1)]
            content=self.get_summarize(content)
        content_counter=self.get_counter(content)
        return(content_counter)
        
        


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




def get_bigram_counts(token,window_size,mystopword):
    stop_word = mystopword.stop_word
    pair_counts = Counter()
   
    for tokens in token:
        for i in range(len(tokens) - 1):
            for distance in range(1, window_size):
                if i + distance < len(tokens):
                    w1 = tokens[i]
                    w2 = tokens[i + distance]
                    if w1 not in stop_word and w2 not in stop_word:
                        pair_counts[(w1, w2)] += 1
    return(pair_counts)
def word_to_vector(this_year_token,pairwise_with_windows_list,mystopword,window_size):
    this_year_vs=[]
    for tokenized_article in progressbar(this_year_token,prefix="run word to vector"):
        finder = nltk.BigramCollocationFinder.from_words([word for word in tokenized_article if word not in mystopword.stop_word],window_size=window_size)
        this_vs= {key: 0 for key in pairwise_with_windows_list}
        for pair,times in finder.ngram_fd.items():
            if pair in this_vs.keys():
                this_vs[pair]=times
        this_year_vs.append(this_vs)            
    return(this_year_vs)
# class Unigram:
#     def __init__(self,target_counter,other_counter):
#         self.target_counter = target_counter
#         self.other_counter = other_counter
        
#     def get_different_corpus_set(self,mystopword,TF_OTHER_THRESHOLD=20,TF_TARGET_THRESHOLD=5):
#         other_corpus_set=set(key for key,times in self.other_counter.items() if times>TF_OTHER_THRESHOLD)-mystopword.stop_word
#         target_corpus_set=set(key for key,times in self.target_counter.items() if times>TF_TARGET_THRESHOLD)-mystopword.stop_word
#         self.different_corpus_set = target_corpus_set-other_corpus_set


# # set var

# In[160]:


TRAIN_START_DATE = "2017-12"

TRAIN_INTIVAL = 8
TEST_INTIVAL = 1


from_date = TRAIN_START_DATE
to_date = str(np.datetime64(TRAIN_START_DATE) +
              np.timedelta64(TRAIN_INTIVAL, 'M'))
test_from_date = str(np.datetime64(to_date) +
              np.timedelta64(1, 'M'))
test_to_date = str(np.datetime64(test_from_date)+np.timedelta64(TEST_INTIVAL, 'M'))

print(from_date,to_date,test_from_date,test_to_date)
window_size=5


# # 定義資料來源

# In[161]:


raw_df = pd.read_csv("../../data/crawler_news_data/oilprice_news.csv")
raw_df_cnbc = pd.read_csv("../../data/crawler_news_data/cnbc_oil_news.csv")
bdate=pd.bdate_range("2009","2019")

raw_df.publish_datetime=pd.DatetimeIndex(raw_df.publish_datetime)
raw_df.loc[~raw_df.publish_datetime.isin(bdate),'publish_datetime']=np.nan
raw_df.publish_datetime=raw_df.publish_datetime.fillna(method='ffill')

raw_df_cnbc.story_publish_datetime=pd.DatetimeIndex(raw_df_cnbc.story_publish_datetime)
raw_df_cnbc.loc[~raw_df_cnbc.story_publish_datetime.isin(bdate),'story_publish_datetime']=np.nan
raw_df_cnbc.story_publish_datetime=raw_df_cnbc.story_publish_datetime.fillna(method='ffill')

data_df=raw_df.sort_values(by="publish_datetime",ascending=True).set_index('publish_datetime')
data_df_cnbc = raw_df_cnbc.sort_values(by="story_publish_datetime",ascending=True).set_index('story_publish_datetime')
data_df_oilprice = pd.DataFrame({"date":raw_df.publish_datetime,"content":raw_df.content})
data_df_cnbc = pd.DataFrame({"date":raw_df_cnbc.story_publish_datetime,"content":raw_df_cnbc.story_full_article})
data_df_oilprice_cnbc = data_df_oilprice.append(data_df_cnbc)
data_df_oilprice_cnbc = data_df_oilprice_cnbc.sort_values(by="date",ascending=True).set_index('date')
raw_content = get_content(data_df_oilprice_cnbc)
raw_content=raw_content.loc[raw_content.apply(clean_text_by_sentences).apply(list).apply(len).apply(lambda x:x>1)]
train_content=raw_content[from_date:to_date]
test_content=raw_content[test_from_date:test_to_date]


# # use sentiment tag effective date

# ## get effective date news date

# In[162]:


effective_news_df = pd.read_csv("../../data/crude_oil_price/effective_news_date_percentage_positive.csv")
effective_news_df.date=pd.DatetimeIndex(effective_news_df.date)
effective_news_df=effective_news_df.set_index('date')
effective_news_date = effective_news_df.index
effective_news_df.head()


# ## load sentiment dict

# In[163]:


sentiment_df = pd.read_csv("../../data/sentiment_dictionary/LoughranMcDonald_MasterDictionary_2016.csv")
pos_dict = pd.read_csv("../../data/sentiment_dictionary/positive-words.txt",sep="\n")
neg_dict = pd.read_csv("../../data/sentiment_dictionary/negative-words.txt",sep="\n")
negative_list = set([str.lower(word) for word in sentiment_df.loc[sentiment_df.Negative !=0].Word]) | set(neg_dict.word)
positive_list = set([str.lower(word) for word in sentiment_df.loc[sentiment_df.Positive !=0].Word]) | set(pos_dict.word)


# ## tag with sentiment

# In[164]:


def tag_with_sentiment(content):
    sid = SentimentIntensityAnalyzer()
    tag_answer=pd.Series(index=content.index)
    for i in range(len(content)):
        today=content.index[i]
        if today in effective_news_df.index :
            pct_change = effective_news_df.two_day_percentage.loc[today]
            try:
                neg = sum(pd.Series(word_tokenize(content[i])).isin(negative_list))
                pos = sum(pd.Series(word_tokenize(content[i])).isin(positive_list))
                if (pct_change > 0 and pos>neg): #or (pct_change < 0 and pos<neg):
                    tag_answer[i]=1
                else:
                    tag_answer[i]=0
            except Exception as e:
                print(str(e))
                tag_answer[i]=0
        else:
            tag_answer[i]=0
    return(tag_answer)


# # prework

# ## 建立stopword

# In[165]:


preprocesser=Preprocessor()


# In[150]:


all_year_counter = preprocesser.preprocess(content=raw_content)
mystopword=MyStopWord(content_counter=all_year_counter,most_common=100)
mystopword.define('c')
mystopword.keep('demand')


# ## find dictionary

# In[166]:


train_content=preprocesser.get_summarize(train_content)

# train_content = summarized_content[from_date:to_date]
train_tokenized_content=preprocesser.get_tokenized_content(train_content)

# target_content = train_content.loc[train_content.index.isin(effective_news_date.values)]
# target_tokenized_content=preprocesser.get_tokenized_content(target_content)

# other_content = train_content.loc[~train_content.index.isin(effective_news_date.values)]
# other_tokenized_content=preprocesser.get_tokenized_content(other_content)


# ### 用Target corpus - Other corpus find bigram dictionary

# In[332]:


# TF_TARGET_THRESHOLD=2
# OTHER_TARGET_THRESHOLD=20

# target_bigram_count=get_bigram_counts(target_tokenized_content,window_size=window_size,mystopword=mystopword)
# other_bigram_count=get_bigram_counts(other_tokenized_content,window_size=window_size,mystopword=mystopword)

# target_corpus_set=set([key for key,times in target_bigram_count.most_common(1000) if times>TF_TARGET_THRESHOLD])
# other_corpus_set=set([key for key,times in other_bigram_count.items() if times>OTHER_TARGET_THRESHOLD])

# pairwise_dictionary = target_corpus_set - other_corpus_set
# print("len(pairwise_dictionary)",len(pairwise_dictionary))


# ### train data find bigram dictionary

# In[167]:


train_bigram_count=get_bigram_counts(train_tokenized_content,window_size=window_size,mystopword=mystopword)
pairwise_dictionary=[key for key , times in train_bigram_count.most_common(400)]
len(pairwise_dictionary)


# # get train_vs

# ## word to vector

# In[168]:


train_vs = word_to_vector(train_tokenized_content,pairwise_dictionary,mystopword,window_size)
train_vs_df=pd.DataFrame(train_vs)
train_vs_df=train_vs_df.set_index(pd.DatetimeIndex(train_content.index))


# In[169]:


print("target:",train_vs_df.loc[effective_news_date.values].sum(axis=1).mean())
print("other:",train_vs_df.loc[~train_content.index.isin(effective_news_date.values)].sum(axis=1).mean())


# In[170]:


train_ans=tag_with_sentiment(train_content)
train_vs_df['tags'] = train_ans
# train_vs_df.loc[train_vs_df.index.isin(effective_news_date.values),'tags']=1
train_vs_df.to_csv("../../data/train_test_dataset/oilprice_cnbc_new_train.csv")


# # Get Test vs

# ## word to vector

# In[176]:


# test_content = preprocesser.get_summarize(test_content)
# test_tokenized_content = preprocesser.get_tokenized_content(content=test_content)
# test_vs = word_to_vector(test_tokenized_content,pairwise_dictionary,mystopword,window_size)
# test_vs_df=pd.DataFrame(test_vs)
# test_vs_df=test_vs_df.set_index(pd.DatetimeIndex(test_content.index))

test_ans=tag_with_sentiment(test_content)
test_vs_df['tags'] = test_ans
# test_vs_df['tags'] = 0
# test_vs_df.loc[test_vs_df.index.isin(effective_news_date.values),'tags'] =1
test_vs_df.to_csv("../../data/train_test_dataset/oilprice_cnbc_new_test.csv")


# In[177]:


sum(test_vs_df.tags)


# In[172]:


print("target:",test_vs_df.loc[test_content.index.isin(effective_news_date.values)].sum(axis=1).mean())
print("other:",test_vs_df.loc[~test_content.index.isin(effective_news_date.values)].sum(axis=1).mean())
print(len(pairwise_dictionary))


# ## generate by Tfidf

# In[314]:


tfidf=TfidfVectorizer()


# In[317]:


tfidf.fit(summarized_content)
test_tfidf_vs=tfidf.transform(test_content)
train_tfidf_vs=tfidf.transform(train_content)


# In[325]:


train_tfidf_vs_df=pd.DataFrame(train_tfidf_vs.toarray())
train_tfidf_vs_df=train_tfidf_vs_df.set_index(pd.DatetimeIndex(train_content.index))
train_tfidf_vs_df['tags'] = 0
train_tfidf_vs_df.loc[train_tfidf_vs_df.index.isin(effective_news_date.values),'tags']=1
train_tfidf_vs_df.to_csv("../../data/train_test_dataset/oilprice_cnbc_new_train.csv")

test_tfidf_vs_df=pd.DataFrame(test_tfidf_vs.toarray())
test_tfidf_vs_df=test_tfidf_vs_df.set_index(pd.DatetimeIndex(test_content.index))
test_tfidf_vs_df['tags'] = 0
test_tfidf_vs_df.loc[test_tfidf_vs_df.index.isin(effective_news_date.values),'tags'] =1
test_tfidf_vs_df.to_csv("../../data/train_test_dataset/oilprice_cnbc_new_test.csv")

