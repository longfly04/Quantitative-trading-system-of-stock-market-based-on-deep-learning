import os,sys

sys.path.insert(0, 'D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning')

from utils.tushare_util import MinuteDownloader


def main():
    """"""
    minite_downloader = MinuteDownloader(stock_code='300672.SC',start_date='20170301', end_date='20200301')
    data = minite_downloader.downloadMinutes()
    pass

if __name__=='__main__':
    main()