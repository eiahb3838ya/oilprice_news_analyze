import nltk
import re
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pprint import pprint
from collections import Counter
from math import *
from datetime import datetime
from progressbar import progressbar
import matplotlib.pyplot as plt


def word_to_vector(this_year_token, pairwise_with_windows_list, mystopword, window_size):
    this_year_vs = []
    for tokenized_article in progressbar(this_year_token, prefix="word to vector"):
        finder = nltk.BigramCollocationFinder.from_words(
            [word for word in tokenized_article if word not in mystopword.stop_word], window_size=window_size)
        this_vs = {key: 0 for key in pairwise_with_windows_list}
        for pair, times in finder.ngram_fd.items():
            if pair in this_vs.keys():
                this_vs[pair] = times
        this_year_vs.append(this_vs)
    return(this_year_vs)


def get_content(data_df):
    content = data_df.content
    content.index = pd.DatetimeIndex(content.index)
    content = content.dropna(how="any")
    return(content)


class Preprocessor:
    def __init__(self, path="", content=False):
        if len(path) > 1:
            raw_df = pd.read_csv(path)
            self.data_df = raw_df.sort_values(
                by="publish_datetime", ascending=True).set_index('publish_datetime')
            content = self.data_df.content
            content.index = pd.DatetimeIndex(content.index)
            content = content.dropna(how="any")
            self.content = content
        else:
            self.content = content

    def stem_and_other_stuff(self, each_news):
        ps = PorterStemmer()
        return([ps.stem(word.lower()) for word in each_news if word.isalpha()])

    def check_alpha_tolower(self, each_news):
        return([word.lower() for word in each_news if word.isalpha()])

    def get_content_from_date(self, from_date, to_date):
        self.content = self.content[from_date:to_date]

    def to_counter(self, stem=True):
        self.token_content = self.content.apply(word_tokenize)
        if stem:
            self.tokens = self.token_content.apply(self.stem_and_other_stuff)
        else:
            self.tokens = self.token_content.apply(self.check_alpha_tolower)
        content_counter = Counter()
        for news in progressbar(self.tokens, prefix="to counter"):
            content_counter.update(news)
        self.counter = content_counter


class MyStopWord:
    def __init__(self, content_counter, most_common=100, stop_word=None):
        from nltk.corpus import stopwords
        self.counter_stop_word = [
            word for word, time in content_counter.most_common(most_common)]
        self.user_keep = []
        self.user_define = []
        if stop_word:
            self.stop_word = stop_word
        else:
            self.stop_word = set(self.counter_stop_word +
                                 stopwords.words('english'))

    def keep(self, word):
        self.user_keep.append(word)
        self.stop_word.discard(word)

    def define(self, word):
        self.user_define.append(word)
        self.stop_word.add(word)


class Unigram:
    def __init__(self, target_counter, other_counter):
        self.target_counter = target_counter
        self.other_counter = other_counter

    def get_different_corpus_set(self, mystopword, TF_OTHER_THRESHOLD=20, TF_TARGET_THRESHOLD=5):
        other_corpus_set = set(key for key, times in self.other_counter.items(
        ) if times > TF_OTHER_THRESHOLD)-mystopword.stop_word
        target_corpus_set = set(key for key, times in self.target_counter.items(
        ) if times > TF_TARGET_THRESHOLD)-mystopword.stop_word
        self.different_corpus_set = target_corpus_set-other_corpus_set


class Bigram:
    def __init__(self, token):
        self.token = token

    def count_word_pair_with_windows(self, window_size, mystopword):
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


# def word_to_vector(this_year_token,pairwise_with_windows_list,mystopword,window_size):
#     this_year_vs=[]
#     for tokenized_article in this_year_token:
#         finder = nltk.BigramCollocationFinder.from_words([word for word in tokenized_article if word not in mystopword.stop_word],window_size=window_size)
#         this_vs= {key: 0 for key in pairwise_with_windows_list}
#         for pair,times in finder.ngram_fd.items():
#             if pair in this_vs.keys():
#                 this_vs[pair]=times
#         this_year_vs.append(this_vs)
#     return(this_year_vs)
