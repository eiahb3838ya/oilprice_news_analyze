#!/usr/bin/env python
# coding: utf-8

# In[125]:


import bs4,requests,urllib3
from datetime import datetime
from ipywidgets import IntProgress
from IPython.display import display
import pandas as pd


# In[110]:


root="https://www.cnbc.com/"


# In[130]:


output_list=[]
page=1
while(True):
    current_page_link="https://www.cnbc.com/oil/?page="+str(page)
    res = requests.get(current_page_link)
    page_soup=bs4.BeautifulSoup(res.text)
    bigHeader_soup=page_soup.find("div",{"class":"stories-lineup bigHeader"})
    if bigHeader_soup:
        # progress = IntProgress()
        # progress.max = len(bigHeader_soup.findAll("div",{"class":"headline"}))
        # progress.description = '(Init)'
        # display(progress)
        # article_progress=1
        for div in bigHeader_soup.findAll("div",{"class":"headline"}):
            # progress.value += 1
            # progress.description = str(round(article_progress/progress.max*100))+"%"
            # article_progress=article_progress+1
            href=div.a['href']
            if not href.startswith("/video"):
                story_publish_datetime=datetime.strptime(href[1:11],"%Y/%m/%d").date()
#                 print(publish_datetime)
                fulllink=root+href
                story_res=requests.get(fulllink)
                story_soup=bs4.BeautifulSoup(story_res.text)
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
    #                 print(story_abstract)
                    story={"story_publish_datetime":story_publish_datetime,
                          "story_title":story_title,
                          "story_full_article":story_full_article,
                          "story_abstract":story_abstract}
                    output_list.append(story)
        print("page #",page,"is done")
        page=page+1
    else:
        print("the crawler is done")
        break

        


# In[132]:


output_df=pd.DataFrame(output_list).set_index("story_publish_datetime")
output_df.to_csv("cnbc_oil_news.csv")

