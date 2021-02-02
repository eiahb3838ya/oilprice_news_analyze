#!/usr/bin/env python
# coding: utf-8

# In[117]:


import bs4,requests,urllib3
from datetime import datetime
from ipywidgets import IntProgress
from IPython.display import display
import pandas as pd
from progress.bar import Bar


# In[93]:


root="https://oilprice.com/Energy/Crude-Oil/"
output_list=[]


# In[122]:


def get_story_content(href):
    story_res=requests.get(href)
    story_soup=bs4.BeautifulSoup(story_res.text)
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


for page in range(1,300):
    if page==1:
        page_res=requests.get(root,timeout=5)
    else:
        page_res=requests.get(root+"Page-"+str(page)+".html")
    print(page,page_res)
    page_soup=bs4.BeautifulSoup(page_res.text)
    bar = Bar('   Processing', max=len(page_soup.findAll("div",{"class":"categoryArticle__content"})))
    for categoryArticle__content in page_soup.findAll("div",{"class":"categoryArticle__content"}):
        story=get_story(categoryArticle__content)
        bar.next()
        output_list.append(story)
    bar.finish()
    print("page #",page,"is done")


# In[109]:


output_df=pd.DataFrame(output_list).set_index("publish_datetime")
output_df.to_csv("oilprice_news.csv")


# In[127]:


get_ipython().system('jupyter nbconvert --to script oilprice_news_crawler.ipynb')

