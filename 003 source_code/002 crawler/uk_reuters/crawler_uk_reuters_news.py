# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:02:08 2019

@author: eiahb
"""
import bs4,requests,os
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

try:
    START_DATE=sys.argv[1]
except:
    START_DATE="2019-02-24"
print("start with ",START_DATE)
try:
    ROOT=sys.argv[2]
except:
    ROOT='C:\\Users\\Evan\\MyFile\\Fortune-street\\007 oil_price\\oilprice_news_analyze\\003 data/002 corpus_data/'
print("save at root : ",ROOT)

root_web="https://uk.reuters.com"
oilRpt_web="https://uk.reuters.com/news/archive/oilRpt?view=page&page="
root_dir=ROOT+"raw_csv_data"





def get_story_content(href):
    story_res=requests.get(href)
    story_soup=bs4.BeautifulSoup(story_res.text,'html.parser')
    full_content=""
    if story_soup:
        content_div=story_soup.find("div",{"class":"StandardArticleBody_body"})
        if content_div:
            for p in content_div.findAll("p"):
                full_content=full_content+p.text
    return(full_content)
    
    
def get_story(story_content):
    try:
        publish_timestr=story_content.time.span.text
        publish_datetime=datetime.strptime(publish_timestr,"%d %b %Y")
    except:
        publish_datetime=datetime.today().date()
        publish_timestr=str(publish_datetime)
        
    href=root_web+story_content.a['href']
    title=str.strip(story_content.h3.text)
    summarize=""
    if (story_content.p):
        summarize=story_content.p.text
    
    content=get_story_content(href)
    story={"publish_datetime":publish_datetime,
          "publish_timestr":publish_timestr,
          "href":href,
          "title":title,
          "summarize":summarize,
          "content":content,
          }
    return(story)

if not os.path.exists(root_dir+"/uk_reuters_news_data"):
    os.makedirs(root_dir+"/uk_reuters_news_data")
if not os.path.exists(root_dir+"/waiting_preprocess/uk_reuters_news_data"):
    os.makedirs(root_dir+"/waiting_preprocess/uk_reuters_news_data")

output_list=[]
latest_date = None
this_date = None
page=1425
for page in range(50000):
    page_res=requests.get(oilRpt_web+str(page))
    print("start process page:",page,page_res)
    page_soup=bs4.BeautifulSoup(page_res.text,'html.parser')
    news_headline_list=page_soup.find("div",{"class":"news-headline-list"})
    story_content_list=news_headline_list.findAll("div",{"class":"story-content"})
#    print(latest_date,this_date)
    for categoryArticle__content in tqdm(story_content_list):
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
            if os.path.exists(root_dir+"/uk_reuters_news_data/"+"uk_reuters_news_"+d+".csv"):
                output_df.to_csv(root_dir+"/uk_reuters_news_data/"+"uk_reuters_news_"+d+".csv",mode='a',header=False)
                output_df.to_csv(root_dir+"/waiting_preprocess/uk_reuters_news_data/"+"uk_reuters_news_"+d+".csv",mode='a',header=False)
            else:
                output_df.to_csv(root_dir+"/uk_reuters_news_data/"+"uk_reuters_news_"+d+".csv")
                output_df.to_csv(root_dir+"/waiting_preprocess/uk_reuters_news_data/"+"uk_reuters_news_"+d+".csv")
            
            #use logging later
            print("date #",d,"has been saved")
            #use logging later
            
            latest_date = this_date
            output_list=[]
            output_list.append(story)
    #get_story(story_content)
