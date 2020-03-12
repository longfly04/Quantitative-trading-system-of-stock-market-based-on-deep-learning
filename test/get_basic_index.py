import os,sys
sys.path.insert(0, 'D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning')

import arrow
import pandas as pd 

from utils.base.stock import IndexData, Parameters


def main():
    """
    下载上证综指，作为参考指标
    """
    start_date = arrow.get('20140601','YYYYMMDD')
    end_date = arrow.now()
    # 目前只提供上证综指，深证成指，上证50，中证500，中小板指，创业板指
    codes = ['000001.SH','399001.SZ','000016.SH','000905.SH','399005.SZ','399006.SZ']
    paralist = []
    
    for ts in codes:
        para = Parameters(  ts_code=ts,
                            start_date=start_date.format('YYYYMMDD'),
                            end_date=end_date.format('YYYYMMDD')
                            )

        index_api = IndexData(pro=para.pro, para=para)
        data = index_api.getIndexDaily()

        total_df = data.sort_values(by=['trade_date'], ascending=True)

        save_path = os.path.join(sys.path[0], 'output')
        total_df.to_csv(os.path.join(save_path, 'index_' + ts + '.csv'))


if __name__=='__main__':
    main()