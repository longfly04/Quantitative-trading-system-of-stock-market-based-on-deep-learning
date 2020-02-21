import pandas as pd
import sys
from utils.tushare_util import *

def main():
    with open('dataset\\上证50成分股.txt', encoding='UTF-8') as f:
        stock_dict = {}
        for line in f.readlines():
            if line == '\n':
                break
            l = list(line.rstrip('\n').split())
            stock_dict[l[0]] = l[1][1:-1]

    for k,v in stock_dict.items():
        # md = MinuteDownloader(start_date='20190101', end_date='20191231', stock_code=str(v))
        # minutes_data = md.downloadMinutes(save=False)
        dd = DailyDownloader(start_date='20090101', end_date='20191231', stock_code=str(v))
        daily_data = dd.downloadDaily(save=True)

        print('[INFO] Complete %s downloading.' %k)


if __name__ == "__main__":
    main()