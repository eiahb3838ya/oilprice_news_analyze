from google_api_downloader import google_api_downloader as gd
from datetime import date, timedelta
# from pprint import pprint
import json,os
import datetime
import logging
import numpy as np
import pandas as pd
from newsapi.newsapi_exception import  NewsAPIException

ROOT="D:/work/school/future_news_analyze/data/google_news_api_data"

# 基礎設定
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[logging.FileHandler('log/news_update.log', 'a', 'utf-8'), ])

# 定義 handler 輸出 sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# 設定輸出格式
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# handler 設定輸出格式
console.setFormatter(formatter)
# 加入 hander 到 root logger
logging.getLogger('').addHandler(console)
logging.info("start our program")
logging.warning("love bebe <3")


with open('google_news_update_record.json', 'r') as f:
    data = json.load(f)
record = data['update_date']
commodity = list()
query = list()
last_update_date = list()
for mission in record:
    commodity.append(mission['commodity'])
    query.append(mission['query'])
    last_update_date.append(mission['last_update_date'])

for date in pd.date_range(start=min(last_update_date), end=datetime.date.today()).date:
    for i in range(0, len(commodity)):
        # init class
        my_gd = gd()

        # get data by date and query
        try:
            temp_news = my_gd.daily_news_data(query[i], str(date))
        except NewsAPIException:
            logging.debug("the download ended because of daily limit,stoped at "+query[i]+" "+str(date))
            raise
        except Exception as err:
            logging.error(err)
            

        temp_news=my_gd.keep_relevacny_sort(temp_news)
        # check folder Exist
        folder_name=ROOT+"/"+commodity[i]
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        # save data
        filename = commodity[i]+"_"+date.strftime('%Y_%m_%d')
        if not os.path.exists(folder_name+"/csv"):
            os.mkdir(folder_name+"/csv")
        temp_news.to_csv(folder_name+"/csv/"+filename+".csv")
        if not os.path.exists(folder_name+"/json"):
            os.mkdir(folder_name+"/json")
        temp_news.to_json(folder_name+"/json/"+filename+".json",orient="records")



        # update logging
        logging.info("Downloaded file:"+filename)

        # update work record json file
        record[i]["last_update_date"] = str(date)
        data["update_date"] = record
        with open('google_news_update_record.json', 'w') as f:
            json.dump(data, f)
