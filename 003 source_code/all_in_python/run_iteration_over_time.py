import pandas as pd
import os


from_date_range = pd.date_range(
    start="2017-01-01", periods=11, freq="MS", closed='left', dtype='datetime64[m]')
for from_date in from_date_range:
    from_date_str=pd.to_datetime(from_date).strftime("%Y-%m")
    os.system('python get_bigram_and_predict.py 5 '+from_date_str)
