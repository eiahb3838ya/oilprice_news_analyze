
# coding: utf-8

# In[312]:

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


# In[248]:

# In[3]:
window_size = 5
TRAIN_START_DATE = "2018-01"

TRAIN_INTIVAL = 8
TEST_INTIVAL = 2

TF_TARGET_THRESHOLD=2
OTHER_TARGET_THRESHOLD=20
DATA_ROOT = "../../../data/"
SUMMARIZE = True
TRAIN_SENTIMENT = False
TEST_SENTIMENT = False

most_common_count = 1000
summarize_word_count = 300

effective_date_path = DATA_ROOT+"crude_oil_price/effective_news_date_days_before_and_after.csv"
from_date = TRAIN_START_DATE
to_date = str(np.datetime64(TRAIN_START_DATE) +
              np.timedelta64(TRAIN_INTIVAL-1, 'M'))

test_from_date = str(np.datetime64(TRAIN_START_DATE) +
              np.timedelta64(TRAIN_INTIVAL, 'M'))
test_to_date = str(np.datetime64(test_from_date)+np.timedelta64(TEST_INTIVAL-1, 'M'))

train_filename = "from_date"+from_date+"_to_date"+to_date+"_test_from_date"+test_from_date+"_test_to_date"+test_to_date


print("########################\n"+"from_date", from_date, "\nto_date", to_date, "\ntest_from_date",
      test_from_date, "\ntest_to_date", test_to_date+"\n########################")

def train(train_documents,vector_size=300):
    model = Doc2Vec(train_documents, vector_size=vector_size, window=4, min_count=2, workers=12)
    model.train(train_documents,total_examples=model.corpus_count,epochs=30)
    return(model)
def get_content(data_df):
    content=data_df.content
    content.index = pd.DatetimeIndex(content.index)
    content=content.dropna(how="any")
    return(content)
class Preprocessor:
    def __init__(self,stopword=[],use_stem=False,use_summarize=True,summarize_word_count=summarize_word_count):
        self.use_stem=use_stem
        self.use_summarize=use_summarize
        self.summarize_word_count=summarize_word_count
        self.stopword=stopword
    def stem_and_other_stuff(self,each_news):
        ps=PorterStemmer()
        return([ps.stem(word.lower()) for word in each_news.split(" ") if word.isalpha() and word not in self.stopword])
    
    def check_alpha_tolower(self,each_news):
        # return([word.lower() for word in each_news.split(" ") if word.isalpha()])
        return([word.lower() for word in each_news if word.isalpha()])
        
    def get_tokenized_content(self,content):
        tokenized_content_s=content.apply(word_tokenize)
        if self.use_stem:        
            output_token=tokenized_content_s.apply(self.stem_and_other_stuff)
        else:
            output_token=tokenized_content_s.apply(self.check_alpha_tolower)
        return(output_token)
    
    def get_counter(self,content):
        tokenized_content_s=self.get_tokenized_content(content)
        content_counter=Counter()
        for aStemmed_token in tokenized_content_s:
            content_counter.update(aStemmed_token)
#             self.counter = content_counter
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
#         self.different_corpus_set = target_corpus_set-other_corpus_setsentiment_df = pd.read_csv("../../data/sentiment_dictionary/LoughranMcDonald_MasterDictionary_2016.csv")

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

# ## 定義資料來源

# In[326]:


# raw_df = pd.read_csv(DATA_ROOT+"crawler_news_data/oilprice_news.csv")
# raw_df_cnbc = pd.read_csv(DATA_ROOT+"crawler_news_data/cnbc_oil_news.csv")
# bdate=pd.bdate_range("2009","2019")

# raw_df.publish_datetime=pd.DatetimeIndex(raw_df.publish_datetime)
# raw_df.loc[~raw_df.publish_datetime.isin(bdate),'publish_datetime']=np.nan
# raw_df.publish_datetime=raw_df.publish_datetime.fillna(method='ffill')

# raw_df_cnbc.story_publish_datetime=pd.DatetimeIndex(raw_df_cnbc.story_publish_datetime)
# raw_df_cnbc.loc[~raw_df_cnbc.story_publish_datetime.isin(bdate),'story_publish_datetime']=np.nan
# raw_df_cnbc.story_publish_datetime=raw_df_cnbc.story_publish_datetime.fillna(method='ffill')

# data_df=raw_df.sort_values(by="publish_datetime",ascending=True).set_index('publish_datetime')
# data_df_cnbc = raw_df_cnbc.sort_values(by="story_publish_datetime",ascending=True).set_index('story_publish_datetime')
# data_df_oilprice = pd.DataFrame({"date":raw_df.publish_datetime,"content":raw_df.content})
# data_df_cnbc = pd.DataFrame({"date":raw_df_cnbc.story_publish_datetime,"content":raw_df_cnbc.story_full_article})
# data_df_oilprice_cnbc = data_df_oilprice.append(data_df_cnbc)
# data_df_oilprice_cnbc = data_df_oilprice_cnbc.sort_values(by="date",ascending=True).set_index('date')
# raw_content = get_content(data_df_oilprice_cnbc)
# raw_content=raw_content.loc[raw_content.apply(clean_text_by_sentences).apply(list).apply(len).apply(lambda x:x>1)]

raw_content_df = pd.read_csv(DATA_ROOT+"crawler_news_data/oilprice_cnbc_news_clean.csv")
raw_content_df.index = pd.DatetimeIndex(raw_content_df.date)
raw_content = raw_content_df.content

pos_dict = pd.read_csv(DATA_ROOT+"sentiment_dictionary/positive-words.txt",sep="\n")
neg_dict = pd.read_csv(DATA_ROOT+"sentiment_dictionary/negative-words.txt",sep="\n")
sentiment_df = pd.read_csv(DATA_ROOT+"sentiment_dictionary/LoughranMcDonald_MasterDictionary_2016.csv")
negative_list = set([str.lower(word) for word in sentiment_df.loc[sentiment_df.Negative !=0].Word]) | set(neg_dict.word)
positive_list = set([str.lower(word) for word in sentiment_df.loc[sentiment_df.Positive !=0].Word]) | set(pos_dict.word)



preprocesser=Preprocessor()


# ## 建立stopword

# In[329]:


all_year_counter = preprocesser.preprocess(content=raw_content)
mystopword=MyStopWord(content_counter=all_year_counter,most_common=100)
mystopword.define('c')
mystopword.keep('demand')


# ## 用Target corpus - Other corpus find dictionary

# ### 1. effectivate news date

# In[330]:


effective_news_df = pd.read_csv(effective_date_path)
effective_news_df.index = pd.DatetimeIndex(effective_news_df.date)
# effective_news_df = pd.read_csv(effective_news_source)
effective_news_date = effective_news_df['date']
effective_news_date=pd.DatetimeIndex(effective_news_date)


# ### 2. find target and other corpus

# In[331]:


if SUMMARIZE:
    # summarized_df = pd.read_csv(DATA_ROOT+"crawler_news_data/oilprice_cnbc_news_summarized.csv")
    # summarized_df.index = pd.DatetimeIndex(summarized_df.date)
    # summarized_content = summarized_df.content
    summarized_content=preprocesser.get_summarize(raw_content)
    train_content = summarized_content[from_date:to_date]
    test_content = summarized_content[test_from_date:test_to_date]
else:
    train_content = raw_content[from_date:to_date]
    test_content = raw_content[test_from_date:test_to_date]
train_tokenized_content=preprocesser.get_tokenized_content(train_content)

target_content = train_content.loc[train_content.index.isin(effective_news_date.values)]
target_tokenized_content=preprocesser.get_tokenized_content(target_content)

other_content = train_content.loc[~train_content.index.isin(effective_news_date.values)]
other_tokenized_content=preprocesser.get_tokenized_content(other_content)


# ### 3. find bigram dictionary

# In[332]:


target_bigram_count=get_bigram_counts(target_tokenized_content,window_size=window_size,mystopword=mystopword)
other_bigram_count=get_bigram_counts(other_tokenized_content,window_size=window_size,mystopword=mystopword)

target_corpus_set=set([key for key,times in target_bigram_count.most_common(most_common_count) if times>TF_TARGET_THRESHOLD])
other_corpus_set=set([key for key,times in other_bigram_count.items() if times>OTHER_TARGET_THRESHOLD])

pairwise_dictionary = target_corpus_set - other_corpus_set
print("len(pairwise_dictionary)",len(pairwise_dictionary))
first_word = [x[0] for x in pairwise_dictionary]
second_word = [x[1] for x in pairwise_dictionary]
dictionary_df = pd.DataFrame({"first":first_word,"second":second_word})
dictionary_df.to_csv(DATA_ROOT+"wordpair_result/summarized_dictionary.csv")


# In[333]:


# train_bigram_count=get_bigram_counts(train_tokenized_content,window_size=window_size,mystopword=mystopword)
# pairwise_dictionary=[key for key , times in train_bigram_count.most_common(400)]
# len(pairwise_dictionary)


# ## 4. word to vector

# In[334]:


train_vs = word_to_vector(train_tokenized_content,pairwise_dictionary,mystopword,window_size)
train_vs_df=pd.DataFrame(train_vs)
train_vs_df=train_vs_df.set_index(pd.DatetimeIndex(train_content.index))


# In[335]:


print("target:",train_vs_df.loc[effective_news_date.values].sum(axis=1).mean())
print("other:",train_vs_df.loc[~train_content.index.isin(effective_news_date.values)].sum(axis=1).mean())


# In[336]:

if TRAIN_SENTIMENT:
    train_ans=tag_with_sentiment(train_content)
    train_vs_df['tags'] = train_ans
else:
    train_vs_df['tags'] = 0
    train_vs_df.loc[train_vs_df.index.isin(effective_news_date.values),'tags']=1
train_vs_df.to_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_summarize_train.csv")


# ## Generate Test Data

# In[337]:

test_tokenized_content = preprocesser.get_tokenized_content(content=test_content)
test_vs = word_to_vector(test_tokenized_content,pairwise_dictionary,mystopword,window_size)
test_vs_df=pd.DataFrame(test_vs)
test_vs_df=test_vs_df.set_index(pd.DatetimeIndex(test_content.index))

if TEST_SENTIMENT:
    test_ans=tag_with_sentiment(test_content)
    test_vs_df['tags'] = test_ans
else:
    test_vs_df['tags'] = 0
    test_vs_df.loc[test_vs_df.index.isin(effective_news_date.values),'tags'] =1
test_vs_df.to_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_summarize_test.csv")


# In[338]:


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
train_tfidf_vs_df.to_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_summarize_tfidf_train.csv")

test_tfidf_vs_df=pd.DataFrame(test_tfidf_vs.toarray())
test_tfidf_vs_df=test_tfidf_vs_df.set_index(pd.DatetimeIndex(test_content.index))
test_tfidf_vs_df['tags'] = 0
test_tfidf_vs_df.loc[test_tfidf_vs_df.index.isin(effective_news_date.values),'tags'] =1
test_tfidf_vs_df.to_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_summarize_tfidf_test.csv")

