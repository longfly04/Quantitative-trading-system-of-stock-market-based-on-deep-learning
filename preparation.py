from utils.data_manage import StockManager, DataDownloader

def prepare_train(config=None, download=False):
    """
    数据准备
    """
    data_cfg = config['data']

    # 初始化数据下载器 更新行情
    data_downloader = DataDownloader(data_path=data_cfg['data_dir'],
                                     stock_list_file=data_cfg['SH50_list_path'],
                                     )
    if download:
        data_downloader.download_stock(download_mode='additional',
                                       start_date=data_cfg['date_range'][0],
                                       date_col=data_cfg['date_col']
                                       )

    trade_calender = data_downloader.get_calender(start_date=data_cfg['date_range'][0])

    stock_mgr = StockManager(data_path=data_cfg['data_dir'],
                            stock_pool=data_cfg['stock_code'],
                            trade_calender=trade_calender,
                            date_col=data_cfg['date_col'],
                            quote_col=data_cfg['daily_quotes'])

    stock_mgr.global_preprocess()
    history = stock_mgr.get_history_data()
    all_quote = stock_mgr.get_quote()
    calender = stock_mgr.get_trade_calender()

    return calender, history, all_quote
