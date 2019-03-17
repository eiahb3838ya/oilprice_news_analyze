#!/usr/bin/env python
# coding: utf-8

# In[125]:


import bs4,requests,urllib3,sys,os
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# In[110]:



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
#ROOT='C:\\Users\\Evan\\MyFile\\Fortune-street\\007 oil_price\\oilprice_news_analyze\\003 data/002 corpus_data'

root_dir=ROOT+"raw_csv_data"
root_web="https://www.cnbc.com/"
def get_story(div):
    try:
        href=div.a['href']
    except:
        print("get href fail")
    if not href.startswith("/video"):
        story_publish_datetime=datetime.strptime(href[1:11],"%Y/%m/%d").date()
        fulllink=root_web+href
        story_title,story_full_article,story_abstract=get_story_content(fulllink)
        story={
            "publish_datetime":story_publish_datetime,
               "href":fulllink,
               "title":story_title,
               "content":story_full_article,
               "story_abstract":story_abstract,
            }
        return(story)
    else :
        return(None)
    
    
def get_story_content(fulllink):
    story_res=requests.get(fulllink)
    story_soup=bs4.BeautifulSoup(story_res.text,'html.parser')
    if story_soup:
        story_title=story_soup.find("h1",{"class":"title"}).text
        story_content_div=story_soup.find("div",{"itemprop":"articleBody"})
        story_full_article=""
        story_abstract=""
        if story_content_div:
            for p in story_content_div.findAll("p"):
                if not p.text.startswith("*"):
                    story_full_article=story_full_article+p.text+" "
                else:
                    story_abstract=story_abstract+p.text+" "
    return(story_title,story_full_article,story_abstract)
    
if not os.path.exists(root_dir+"/cnbc_news_data"):
    os.makedirs(root_dir+"/cnbc_news_data")
if not os.path.exists(root_dir+"/waiting_preprocess/cnbc_news_data"):
    os.makedirs(root_dir+"/waiting_preprocess/cnbc_news_data")
    
# In[130]:

output_list=[]
latest_date = None
page=1
while(True):
    current_page_link="https://www.cnbc.com/oil/?page="+str(page)
    res = requests.get(current_page_link)
    print("start process page:",page,res)
    page_soup=bs4.BeautifulSoup(res.text,'html.parser')
    bigHeader_soup=page_soup.find("div",{"class":"stories-lineup bigHeader"})
    if bigHeader_soup:
        # progress = IntProgress()
        # progress.max = len(bigHeader_soup.findAll("div",{"class":"headline"}))
        # progress.description = '(Init)'
        # display(progress)
        # article_progress=1
        for div in tqdm(bigHeader_soup.findAll("div",{"class":"headline"})):
            story=get_story(div)
            if not story:
                continue;
            this_date=np.datetime64(story['publish_datetime'])
            if not latest_date :
                latest_date = this_date
            if this_date == latest_date:
                output_list.append(story)
            else:
                ts = pd.to_datetime(str(latest_date)) 
                d = ts.strftime('%Y_%m_%d')
                output_df=pd.DataFrame(output_list).set_index("publish_datetime")
                if os.path.exists(root_dir+"/cnbc_news_data/"+"cnbc_news_"+d+".csv"):
                    output_df.to_csv(root_dir+"/cnbc_news_data/"+"cnbc_news_"+d+".csv",mode='a',header=False)
                    output_df.to_csv(root_dir+"/waiting_preprocess/cnbc_news_data/"+"cnbc_news_"+d+".csv",mode='a',header=False)
                else:
                    output_df.to_csv(root_dir+"/cnbc_news_data/"+"cnbc_news_"+d+".csv")
                    output_df.to_csv(root_dir+"/waiting_preprocess/cnbc_news_data/"+"cnbc_news_"+d+".csv")
                
                #use logging later
                print("date #",d,"has been saved")
                #use logging later
                
                if np.datetime64(this_date) < np.datetime64(START_DATE):
                    print("np.datetime64(this_date) < np.datetime64(START_DATE)")
                    sys.exit()
                
                latest_date = this_date
                output_list=[]
                output_list.append(story)
        print("page #",page,"is done")
        page=page+1
    else:
        ts = pd.to_datetime(str(latest_date)) 
        d = ts.strftime('%Y_%m_%d')
        output_df=pd.DataFrame(output_list).set_index("publish_datetime")
        if os.path.exists(root_dir+"/cnbc_news_data/"+"cnbc_news_"+d+".csv"):
            output_df.to_csv(root_dir+"/cnbc_news_data/"+"cnbc_news_"+d+".csv",mode='a',header=False)
            output_df.to_csv(root_dir+"/waiting_preprocess/cnbc_news_data/"+"cnbc_news_"+d+".csv",mode='a',header=False)
        else:
            output_df.to_csv(root_dir+"/cnbc_news_data/"+"cnbc_news_"+d+".csv")
            output_df.to_csv(root_dir+"/waiting_preprocess/cnbc_news_data/"+"cnbc_news_"+d+".csv")        
        #use logging later
        print("date #",d,"has been saved")
        #use logging later
        
        print("not bigHeader_soup")
        sys.exit()

        


# In[132]:



