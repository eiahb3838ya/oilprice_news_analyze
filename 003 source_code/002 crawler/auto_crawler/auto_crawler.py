# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:31:39 2019

@author: eiahb
"""
import os,json, subprocess
from datetime import datetime

START_DATE=""
if not START_DATE:
    with open('startdate_record.json', 'r') as outfile:
        START_DATE=json.load(outfile)["START_DATE"]
        
ROOT_DIR_OF_PROJECT="C:\\Users\\Evan\\MyFile\\Fortune-street\\007 oil_price\\oilprice_news_analyze\\"
ROOT_DIR=ROOT_DIR_OF_PROJECT+"003 source_code/002 crawler"

for root, dirs, files in os.walk(ROOT_DIR):
#    print("路徑：", root)
    print("檔案：", files)
    for filename in files:
        if filename.startswith("crawler"):
            print("activate crawler "+filename)
            target_file=root.replace('\\','/')+"/"+filename
            subprocess.call(["python",target_file,START_DATE])


START_DATE=datetime.today().date()
with open('startdate_record.json', 'w') as outfile:
    json.dump({"START_DATE":str(START_DATE)}, outfile)
