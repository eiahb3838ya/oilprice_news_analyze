
# coding: utf-8

# In[7]:


import bs4,requests,urllib3,sys
from datetime import datetime
from ipywidgets import IntProgress
from IPython.display import display
from progressbar import progressbar
import pandas as pd
# from progress.bar import Bar


# In[143]:


root="https://economictimes.indiatimes.com"
entrance=root+"/topic/crude-oil/news"
iter_root=root+"/topics_all.cms?type=article&query=crude%20oil&curpg="
output_list=[]
TARGET_PAGES=int(sys.argv[1] )


# In[151]:


def get_story(news_div_list,i=0):
    output_list=[]
    for news_div in progressbar(news_div_list,prefix=str(i)+"love bebe"):
        try:
            href=news_div.find("a")['href']
            title=news_div.find("a").find("h3").text
            time_str=news_div.find("time").text
            content=get_story_content(href)
            try: 
                datetime_=datetime.strptime(time_str[:21],"%d %b, %Y, %H.%M%p")
                date_=datetime_.date
            except:
                datetime_=None
                date_=None
            output_list.append({"publish_datetime":datetime_,
                                "publish_date":date_,
                                "publish_datetime_str":time_str,
                                "href":href,
                                "title":title,
                                "content":content
                                })
        except Exception as err:
            print("something went bullshit"+str(err))
    return(output_list)


# In[140]:


def get_story_content(href,root="https://economictimes.indiatimes.com"):
    story_url=root+href
    story_res=requests.get(story_url)
    story_soup=bs4.BeautifulSoup(story_res.text)
    return(story_soup.find("div",{"class":"Normal"}).text)


# In[ ]:


output_list=[]
for i,entrance in enumerate([iter_root+str(i) for i in range(1,TARGET_PAGES)],start=1):
    root_res=requests.get(entrance)
    root_soup=bs4.BeautifulSoup(root_res.text)
    news_div_list=root_soup.findAll("div")
    output_list=output_list+get_story(news_div_list,i)


# In[150]:


output_df=pd.DataFrame(output_list)
output_df.to_csv("economictimes_news_data.csv",index=False)


# In[154]:


get_ipython().system('jupyter nbconvert --to script economictimes_news.ipynb')

