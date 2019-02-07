#!/usr/bin/env python
# coding: utf-8

# In[117]:


import bs4,requests,urllib3,os
from datetime import datetime
from ipywidgets import IntProgress
from IPython.display import display
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys



# In[93]:
try:
    START_DATE=sys.argv[1]
except:
    START_DATE="2018-01-01"
print("start with ",START_DATE)
try:
    ROOT=sys.argv[2]
except:
    ROOT="D:/work/fortune_street/002 news_analyze/002 data/002 corpus_data/"
print("save at root : ",ROOT)

root_web="https://oilprice.com/Energy/Crude-Oil/"
root_dir=ROOT+"raw_csv_data"



# In[122]:


def get_story_content(href):
    story_res=requests.get(href)
    story_soup=bs4.BeautifulSoup(story_res.text,'html.parser')
    full_content=""
    if story_soup:
        content_div=story_soup.find("div",{"id":"article-content"})
        if content_div:
            for p in content_div.findAll("p"):
                full_content=full_content+p.text
    return(full_content)


# In[123]:


def get_story(categoryArticle__content):
    try:
        meta=categoryArticle__content.find("p",{"class":"categoryArticle__meta"})
    except:
        meta=None
        print(categoryArticle__content)
    if meta:
        meta=meta.text
        publish_timestr=meta.split("|")[0][:12]
    try:
        publish_datetime=datetime.strptime(publish_timestr,"%b %d, %Y")
    except:
        publish_datetime=None
    href=categoryArticle__content.a['href']
    title=categoryArticle__content.h2.text
    content=get_story_content(href)
    story={"publish_datetime":publish_datetime,
          "publish_timestr":publish_timestr,
          "meta":meta,
          "href":href,
          "title":title,
          "content":content}
    return(story)


# In[125]:
if not os.path.exists(root_dir+"/oil_price_news_data"):
    os.makedirs(root_dir+"/oil_price_news_data")
if not os.path.exists(root_dir+"/waiting_preprocess/oil_price_news_data"):
    os.makedirs(root_dir+"/waiting_preprocess/oil_price_news_data")
    

output_list=[]
latest_date = None

for page in range(1,300):
    if page==1:
        page_res=requests.get(root_web,timeout=5)
    else:
        page_res=requests.get(root_web+"Page-"+str(page)+".html")
    print("start process page:",page,page_res)
    page_soup=bs4.BeautifulSoup(page_res.text,'html.parser')
#    bar = Bar('   Processing', max=len(page_soup.findAll("div",{"class":"categoryArticle__content"})))
    for categoryArticle__content in tqdm(page_soup.findAll("div",{"class":"categoryArticle__content"})):
        story=get_story(categoryArticle__content)
        this_date=np.datetime64(story['publish_datetime'])
        if not latest_date :
            latest_date = this_date
        if this_date == latest_date:
            output_list.append(story)
        else:
            if np.datetime64(this_date) < np.datetime64(START_DATE):
                sys.exit()
            ts = pd.to_datetime(str(latest_date)) 
            d = ts.strftime('%Y_%m_%d')
            output_df=pd.DataFrame(output_list).set_index("publish_datetime")
            if os.path.exists(root_dir+"/oil_price_news_data/"+"oilprice_news_"+d+".csv"):
                output_df.to_csv(root_dir+"/oil_price_news_data/"+"oilprice_news_"+d+".csv",mode='a',header=False)
                output_df.to_csv(root_dir+"/waiting_preprocess/oil_price_news_data/"+"oilprice_news_"+d+".csv",mode='a',header=False)
            else:
                output_df.to_csv(root_dir+"/oil_price_news_data/"+"oilprice_news_"+d+".csv")
                output_df.to_csv(root_dir+"/waiting_preprocess/oil_price_news_data/"+"oilprice_news_"+d+".csv")
            
            #use logging later
            print("date #",d,"has been saved")
            #use logging later
            
            latest_date = this_date
            output_list=[]
            output_list.append(story)
            
    print("page #",page,"is done")


# In[109]:


#output_df=pd.DataFrame(output_list).set_index("publish_datetime")
#output_df.to_csv("oilprice_news.csv")


# In[127]latest_date:




