# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:31:39 2019

@author: eiahb
"""
import os,json, subprocess
from datetime import datetime

ROOT_DIR_OF_PROJECT="C:\\Users\\Evan\\MyFile\\Fortune-street\\007 oil_price\\oilprice_news_analyze\\"

ROOT_DIR=ROOT_DIR_OF_PROJECT+"003 source_code/002 crawler"
ROOT_FOR_SUBPROCESS=ROOT_DIR_OF_PROJECT+"002 data/002 corpus_data/"
START_DATE=""
if not START_DATE:
    with open(ROOT_DIR+'/auto_crawler/startdate_record.json', 'r') as outfile:
        START_DATE=json.load(outfile)["START_DATE"]
        


for root, dirs, files in os.walk(ROOT_DIR):
#    print("路徑：", root)
    print("檔案：", files)
    for filename in files:
        if filename.startswith("crawler"):
            print("activate crawler "+filename)
            target_file=root.replace('\\','/')+"/"+filename
            print(target_file)
            subprocess.check_output(["python",target_file,START_DATE,ROOT_FOR_SUBPROCESS])



START_DATE=datetime.today().date()
with open('startdate_record.json', 'w') as outfile:
    json.dump({"START_DATE":str(START_DATE)}, outfile)
#C:\Users\Evan\MyFile\Fortune-street\007 oil_price\oilprice_news_analyze\003 source_code\002 crawler\auto_crawler