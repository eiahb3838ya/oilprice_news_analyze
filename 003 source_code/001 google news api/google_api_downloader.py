import requests,datetime,math,json
from newsapi import NewsApiClient

import pandas as pd



class google_api_downloader:
    def __init__(self,key_list=[]):
        self.key_list=key_list+['2ce65575650346a59b89a71e0f937305']
        self.newsapiclient=NewsApiClient(api_key=self.key_list[0])
        
    def get_news_data(self,q,from_param,to=None,page=1,first_request=False):
        if not to :
            to = from_param
        response=self.newsapiclient.get_everything(q=q,
                       from_param=str(from_param),
                       to=str(to),
                       language="en",
                       sort_by='relevancy',
                       page_size=100,page=page)
        if response['status']=='ok':
            page_count_int=min(math.ceil(response['totalResults']/100),10)
            # print(page_count_int)
            out_data_df=pd.DataFrame(response['articles'])
            if first_request:
                return(page_count_int,out_data_df)
            else:
                return(out_data_df)
    def daily_news_data(self,q,day):
        page_count_int,out_data_df=self.get_news_data(q=q,from_param=day,first_request=True)
        for i in range(2,page_count_int+1):
            try:
                out_data_df = pd.concat([out_data_df,self.get_news_data(q=q,from_param=day,page=i)],axis=0,ignore_index=True)
            except Exception as ex:
                # print(ex)
                raise
        return (out_data_df)
    def keep_relevacny_sort (self,df):
        df['relevancy']=df.index.values
        # df=df.set_index(['publishedAt'])
        df=df.sort_values(by="publishedAt").reset_index(drop=True)
        return(df)