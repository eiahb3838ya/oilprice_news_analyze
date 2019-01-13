from google_api_downloader import google_api_downloader as gd
# from pprint import pprint
import json
import datetime
import logging
import numpy as np
import pandas as pd

# 基礎設定
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[logging.FileHandler('my.log', 'a', 'utf-8'), ])

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


with open('google_news_update_record.json', 'r') as f:
    data = json.load(f)
record = data['update_date']
commodity=list()
query=list()
last_update_date=list()
for mission in record:
    commodity.append(mission['commodity'])
    query.append(mission['query'])
    last_update_date.append(mission['last_update_date'])
most_last=min([np.datetime64(day) for day in last_update_date])
for pd.date_range(start,end=datetime.date.today())

