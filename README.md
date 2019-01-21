# 基於新聞對期貨價格的預測與自然語言處裡
## 文件夾說明
### 000 doc
工作過程中的說明文件
### 001 ref 
工作過程中的參考文件

1. paper 
   
   所有參考的論文文章
2. school_report 
   
   畢業專題的完整紀錄
3. web
   
   所有參考的網站
### 002 data
工作過程中的資料
1. bigram_dictionary
   
   用二元詞彙自動建立的字典     
2. google_news_api_data
   
   用 google api 下載的新聞資料關鍵字包含原油黃金與虛擬貨幣
3. news_data
   
   新聞資料，包含原始資料與切詞後的資料，各自以新聞來源做文件夾分類
4. price_data
   
   價格資料，現包含原油期貨
5. school_report_used_data
   
   畢業專題使用的完整資料，會與其他資料有所重疊(參考使用)
6. sentiment_dictionary
   
   網路上建立的情緒字典
7. stop_words
   
   停用詞字典
### 003 source_code
工作過程中的原始碼
1. 000 school report code
   
   畢業專題使用的完整程式碼，會與其他資料有所重疊(參考使用)
2. 001 google news api
    
   下載 google news 的程式碼，調用 google news api
3. 002 crawler
   
   economictimes為台大提供的新聞資料庫，其餘為爬蟲程式新聞來源有
   1. https://oilprice.com/Energy/Crude-Oil/
   2. https://www.cnbc.com
4. 003 preprocess
   
    執行的前處理，(school_report_preprocess 為畢業專題的前處理程式碼參考用)，包含:
   1. preprocess_to_txt_token.py 
        
      切詞，停用詞處理
5. 004 analyze_price
   就價格進行分析，定義自動標籤目標新聞的方式包含:
   1. price_fluctuation_get_effectivedate
   
      基於大波動標籤
   2. price_percentage_get_effectivedate
   
      基於上漲或下跌標籤
6. 999 predict
   
   畢業專題的預測程式碼參考用
7. all_in_python
   
   畢業專題的完整 py 檔程式碼參考用，實現 folding validation


