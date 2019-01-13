
# coding: utf-8

# In[106]:


import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix,consensus_score
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_,f_classif,SelectPercentile,SelectFpr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN
from imblearn.under_sampling import AllKNN


# In[116]:

DATA_ROOT = "../../../data/"
test_df = pd.read_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_test.csv")
train_df = pd.read_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_train.csv")


# In[117]:


x_drop = ['tags','date']
train_x = train_df.drop(x_drop,axis=1).values.astype(float)
train_y = train_df['tags'].values.astype(float)
test_x = test_df.drop(x_drop,axis=1).values.astype(float)
test_y = test_df['tags'].values.astype(float)
print("train_x.shape",train_x.shape)
print("test_x.shape",test_x.shape)


# In[118]:


sm = SMOTE()
train_x, train_y = sm.fit_resample(train_x,train_y)
print("X_res.shape",train_x.shape)
print("y_res.shape",train_y.shape)


# In[119]:


# pca=PCA()
# pca.fit(train_x)
# train_x = pca.transform(train_x)
# test_x = pca.transform(test_x)

selector=SelectFpr(chi2)
train_x=selector.fit_transform(train_x,train_y)
test_x=selector.transform(test_x)

normalizer = Normalizer()
train_x = normalizer.fit_transform(train_x)
test_x = normalizer.transform(test_x)


print("train_x.shape",train_x.shape)
print("test_x.shape",test_x.shape)


# In[120]:


# Perform classification with SVM, kernel=linear 
svc_model = svm.LinearSVC(C=9000) 
svc_model.fit(train_x, train_y) 
prediction = svc_model.predict(test_x)
print("svc_model")
print (classification_report(test_y, prediction,))
print(confusion_matrix(test_y, prediction))

nb_model=MultinomialNB(alpha=0.0001)
nb_model.fit(train_x, train_y) 
prediction = nb_model.predict(test_x)
print("nb_model")
print (classification_report(test_y, prediction,))
print(confusion_matrix(test_y, prediction))

rfc_model=RandomForestClassifier()
rfc_model.fit(train_x, train_y) 
prediction = rfc_model.predict(test_x)
print("rfc_model")
print (classification_report(test_y, prediction,))
print(confusion_matrix(test_y, prediction))


# # ## draw draw see

# # In[236]:


# def word_to_vector(this_year_token,pairwise_with_windows_list,mystopword,window_size):
#     this_year_vs=[]
#     for tokenized_article in progressbar(this_year_token,prefix="word to vector"):
#         finder = nltk.BigramCollocationFinder.from_words([word for word in tokenized_article if word not in mystopword.stop_word],window_size=window_size)
#         this_vs= {key: 0 for key in pairwise_with_windows_list}
#         for pair,times in finder.ngram_fd.items():
#             if pair in this_vs.keys():
#                 this_vs[pair]=times
#         this_year_vs.append(this_vs)            
#     return(this_year_vs)

# def get_content(data_df):
#     content=data_df.content
#     content.index = pd.DatetimeIndex(content.index)
#     content=content.dropna(how="any")
#     return(content)
# class Preprocessor:
#     def __init__(self,path="",content=False):
#         if len(path)>1:
#             raw_df = pd.read_csv(path)
#             self.data_df = raw_df.sort_values(by="publish_datetime",ascending=True).set_index('publish_datetime')
#             content=self.data_df.content
#             content.index = pd.DatetimeIndex(content.index)
#             content=content.dropna(how="any")
#             self.content = content
#         else:
#             self.content = content
#     def stem_and_other_stuff(self,each_news):
#         ps=PorterStemmer()
#         return([ps.stem(word.lower()) for word in each_news if word.isalpha()])
#     def check_alpha_tolower(self,each_news):
#         return([word.lower() for word in each_news if word.isalpha()])
#     def get_content_from_date(self,from_date,to_date):
#         self.content = self.content[from_date:to_date]
#     def to_counter(self,stem=False):
#         self.token_content=self.content.apply(word_tokenize)
#         if stem:        
#             self.tokens=self.token_content.apply(self.stem_and_other_stuff)
#         else:
#             self.tokens=self.token_content.apply(self.check_alpha_tolower)
#         content_counter = Counter()
#         for news in progressbar(self.tokens,prefix="to counter"):
#             content_counter.update(news)
#         self.counter = content_counter


# class MyStopWord:
#     def __init__(self,content_counter,most_common=100,stop_word=None):
#         from nltk.corpus import stopwords
#         self.counter_stop_word=[word for word,time in content_counter.most_common(most_common)]
#         self.user_keep=[]
#         self.user_define=[]
#         if stop_word:
#             self.stop_word=stop_word
#         else:
#             self.stop_word=set(self.counter_stop_word+stopwords.words('english')) 
#     def keep(self,word):
#         self.user_keep.append(word)
#         self.stop_word.discard(word)
#     def define(self,word):
#         self.user_define.append(word)
#         self.stop_word.add(word)

# class Unigram:
#     def __init__(self,target_counter,other_counter):
#         self.target_counter = target_counter
#         self.other_counter = other_counter
        
#     def get_different_corpus_set(self,mystopword,TF_OTHER_THRESHOLD=20,TF_TARGET_THRESHOLD=5):
#         other_corpus_set=set(key for key,times in self.other_counter.items() if times>TF_OTHER_THRESHOLD)-mystopword.stop_word
#         target_corpus_set=set(key for key,times in self.target_counter.items() if times>TF_TARGET_THRESHOLD)-mystopword.stop_word
#         self.different_corpus_set = target_corpus_set-other_corpus_set

# class Bigram:
#     def __init__(self,token):
#         self.token = token
#     def count_word_pair_with_windows(self,window_size,mystopword):
#         stop_word = mystopword.stop_word
#         self.pair_counts = Counter()
#         self.pair_distance_counts = Counter()
#         for tokens in self.token:
#             for i in range(len(tokens) - 1):
#                 for distance in range(1, window_size):
#                     if i + distance < len(tokens):
#                         w1 = tokens[i]
#                         w2 = tokens[i + distance]
#                         if w1 not in stop_word and w2 not in stop_word:
#                             self.pair_distance_counts[(w1, w2, distance)] += 1
#                             self.pair_counts[(w1, w2)] += 1


# # In[237]:



# # raw_content=raw_df.set_index(pd.DatetimeIndex(raw_df.publish_datetime)).content.sort_index()

# TRAIN_START_DATE = "2017-09"

# TRAIN_INTIVAL = 8
# TEST_INTIVAL = 2

# from_date = TRAIN_START_DATE
# to_date = str(np.datetime64(TRAIN_START_DATE) +
#               np.timedelta64(TRAIN_INTIVAL, 'M'))
# test_from_date = str(np.datetime64(to_date) +
#               np.timedelta64(1, 'M'))
# test_to_date = str(np.datetime64(test_from_date)+np.timedelta64(TEST_INTIVAL, 'M'))

# print(from_date,to_date,test_from_date,test_to_date)


# bdate=pd.bdate_range("2009","2019")
# window_size=5
# raw_df = pd.read_csv("../../data/crawler_news_data/oilprice_news.csv")
# raw_df_cnbc = pd.read_csv("../../data/crawler_news_data/cnbc_oil_news.csv")

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
# train_content = raw_content[from_date:to_date]


# # In[238]:


# if len(test_to_date):
#     test_content = raw_content[test_from_date:test_to_date]
# else:
#     test_content = raw_content[test_from_date:]


# # In[239]:


# test_y_date=test_content.loc[test_y.astype(bool)].index.unique()


# # In[240]:


# prediction_date=test_content.loc[prediction.astype(bool)].index.unique()


# # In[241]:


# #recall
# sum(test_y_date.isin(prediction_date))/len(test_y_date)


# # In[242]:


# sum(prediction_date.isin(test_y_date))/len(prediction_date)


# # In[194]:


# effective_vs = test_df.loc[prediction==1]
# non_effective_vs = test_df.loc[prediction!=1]
# pairwise_dictionary = list(test_df.columns[1:-1])
# trace1 = go.Bar(
#     x=pairwise_dictionary,
#     y=[ i/len(effective_vs) for i in list(effective_vs.sum()[1:-1].values)],
#     name='Effective_sum'
# )
# trace2 = go.Bar(
#     x=pairwise_dictionary,
#     y=[ i/len(non_effective_vs) for i in list(non_effective_vs.sum()[1:-1].values)],
#     name='Non_effective_sum'
# )

# data = [trace1, trace2]
# layout = go.Layout(
#     barmode='group'
# )

# fig = go.Figure(data=data, layout=layout)
# iplot(fig, filename='stacked-bar')

