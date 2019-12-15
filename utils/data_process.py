import os
import sys
import pandas as pd
import numpy as np
import math
import datetime
import calendar
import arrow
import time
import matplotlib.pyplot as plt

from .base.stock import *
from .tools import *


class DataProcessor():
    '''
    时序数据处理器，对日线数据和分钟数据进行预处理
    '''
    def __init__(self, config):
        self.cfg = config
        self.data_cfg = config['data']
        self.pre_cfg = config["preprocess"]

    def _choose_color(self, num=1):
        '''
        选个色号吧
        '''
        import random

        cnames = {#'aliceblue':            '#F0F8FF',
                #    'antiquewhite':         '#FAEBD7',
                    'aqua':                 '#00FFFF',
                #    'aquamarine':           '#7FFFD4',
                #    'azure':                '#F0FFFF',
                #    'beige':                '#F5F5DC',
                #    'bisque':               '#FFE4C4',
                    'black':                '#000000',
                #    'blanchedalmond':       '#FFEBCD',
                    'blue':                 '#0000FF',
                    'blueviolet':           '#8A2BE2',
                    'brown':                '#A52A2A',
                    'burlywood':            '#DEB887',
                    'cadetblue':            '#5F9EA0',
                #   'chartreuse':           '#7FFF00',
                    'chocolate':            '#D2691E',
                #    'coral':                '#FF7F50',
                    'cornflowerblue':       '#6495ED',
                #    'cornsilk':             '#FFF8DC',
                    'crimson':              '#DC143C',
                #    'cyan':                 '#00FFFF',
                    'darkblue':             '#00008B',
                    'darkcyan':             '#008B8B',
                    'darkgoldenrod':        '#B8860B',
                    'darkgray':             '#A9A9A9',
                    'darkgreen':            '#006400',
                    'darkkhaki':            '#BDB76B',
                    'darkmagenta':          '#8B008B',
                    'darkolivegreen':       '#556B2F',
                    'darkorange':           '#FF8C00',
                    'darkorchid':           '#9932CC',
                    'darkred':              '#8B0000',
                    'darksalmon':           '#E9967A',
                    'darkseagreen':         '#8FBC8F',
                    'darkslateblue':        '#483D8B',
                    'darkslategray':        '#2F4F4F',
                    'darkturquoise':        '#00CED1',
                    'darkviolet':           '#9400D3',
                    'deeppink':             '#FF1493',
                    'deepskyblue':          '#00BFFF',
                    'dimgray':              '#696969',
                    'dodgerblue':           '#1E90FF',
                    'firebrick':            '#B22222',
                #    'floralwhite':          '#FFFAF0',
                    'forestgreen':          '#228B22',
                #    'fuchsia':              '#FF00FF',
                 #   'gainsboro':            '#DCDCDC',
                 #   'ghostwhite':           '#F8F8FF',
                #    'gold':                 '#FFD700',
                    'goldenrod':            '#DAA520',
                    'gray':                 '#808080',
                    'green':                '#008000',
                    'greenyellow':          '#ADFF2F',
                #    'honeydew':             '#F0FFF0',
                    'hotpink':              '#FF69B4',
                    'indianred':            '#CD5C5C',
                    'indigo':               '#4B0082',
                #    'ivory':                '#FFFFF0',
                    'khaki':                '#F0E68C',
                    'lavender':             '#E6E6FA',
                #    'lavenderblush':        '#FFF0F5',
                    'lawngreen':            '#7CFC00',
                #    'lemonchiffon':         '#FFFACD',
                    'lightblue':            '#ADD8E6',
                #    'lightcoral':           '#F08080',
                #    'lightcyan':            '#E0FFFF',
                #    'lightgoldenrodyellow': '#FAFAD2',
                    'lightgreen':           '#90EE90',
                 #   'lightgray':            '#D3D3D3',
                 #   'lightpink':            '#FFB6C1',
                 #   'lightsalmon':          '#FFA07A',
                    'lightseagreen':        '#20B2AA',
                    'lightskyblue':         '#87CEFA',
                    'lightslategray':       '#778899',
                    'lightsteelblue':       '#B0C4DE',
                #    'lightyellow':          '#FFFFE0',
                    'lime':                 '#00FF00',
                    'limegreen':            '#32CD32',
                 #   'linen':                '#FAF0E6',
                 #   'magenta':              '#FF00FF',
                    'maroon':               '#800000',
                    'mediumaquamarine':     '#66CDAA',
                    'mediumblue':           '#0000CD',
                    'mediumorchid':         '#BA55D3',
                    'mediumpurple':         '#9370DB',
                    'mediumseagreen':       '#3CB371',
                    'mediumslateblue':      '#7B68EE',
                    'mediumspringgreen':    '#00FA9A',
                    'mediumturquoise':      '#48D1CC',
                    'mediumvioletred':      '#C71585',
                    'midnightblue':         '#191970',
                #    'mintcream':            '#F5FFFA',
                #    'mistyrose':            '#FFE4E1',
                #    'moccasin':             '#FFE4B5',
                #    'navajowhite':          '#FFDEAD',
                    'navy':                 '#000080',
                #    'oldlace':              '#FDF5E6',
                    'olive':                '#808000',
                    'olivedrab':            '#6B8E23',
                #    'orange':               '#FFA500',
                #    'orangered':            '#FF4500',
                    'orchid':               '#DA70D6',
                #    'palegoldenrod':        '#EEE8AA',
                    'palegreen':            '#98FB98',
                #    'paleturquoise':        '#AFEEEE',
                    'palevioletred':        '#DB7093',
                #    'papayawhip':           '#FFEFD5',
                #    'peachpuff':            '#FFDAB9',
                    'peru':                 '#CD853F',
                #    'pink':                 '#FFC0CB',
                #    'plum':                 '#DDA0DD',
                #    'powderblue':           '#B0E0E6',
                    'purple':               '#800080',
                    'red':                  '#FF0000',
                    'rosybrown':            '#BC8F8F',
                    'royalblue':            '#4169E1',
                    'saddlebrown':          '#8B4513',
                    'salmon':               '#FA8072',
               #     'sandybrown':           '#FAA460',
                    'seagreen':             '#2E8B57',
                #    'seashell':             '#FFF5EE',
                    'sienna':               '#A0522D',
                #    'silver':               '#C0C0C0',
                    'skyblue':              '#87CEEB',
                    'slateblue':            '#6A5ACD',
                    'slategray':            '#708090',
                #    'snow':                 '#FFFAFA',
                 #   'springgreen':          '#00FF7F',
                    'steelblue':            '#4682B4',
                    'tan':                  '#D2B48C',
                    'teal':                 '#008080',
                #    'thistle':              '#D8BFD8',
                    'tomato':               '#FF6347',
                    'turquoise':            '#40E0D0',
                #    'violet':               '#EE82EE',
                 #   'wheat':                '#F5DEB3',
                #    'white':                '#FFFFFF',
                #    'whitesmoke':           '#F5F5F5',
                #    'yellow':               '#FFFF00',
                #    'yellowgreen':          '#9ACD32'
                }
        if num==1:
            return random.choice(list(cnames.keys()))
        else:
            return [random.choice(list(cnames.keys())) for _ in range(num)]

    @info
    def get_data_statistics(self, data):
        '''
        获取数据的统计信息
        '''

        print('1.数据集共有{}个样本，{}个特征。'.format(data.shape[0], data.shape[1]))
        print('2.数据集基本信息：')
        print(data.describe())
        print('3.数据集包含空值情况统计：')
        print(data.isna().sum())
        print('4.数据集特征和数据类型情况：')
        df = pd.DataFrame({'Data Type':data.dtypes})
        print(df)

    @info
    def drop_dup_fill_nan(self, dataframe):
        '''
        去掉重复的列数据 将空值填上数据 符合时间序列的连续性 输入为股价数据集 输出为去重之后的股价数据集
        '''
        data_new = dataframe.T.drop_duplicates(keep='first').T 
        # 去掉重复的数据列 使用转置再转置的方式 
        data_new.fillna(axis=0, method='ffill', inplace=True)
        # 用之前的值填充空值 确保时间序列的连续性 剩下的空值用0填充
        data_new.fillna(0, inplace=True)
        # 剩下的空值用0填充
        return data_new

    @info
    def fill_inf(self, dataframe):
        '''
        处理数据集的无穷值，用固定值填充，或者用0
        '''
        data_ = np.array(dataframe)
        data_filled = np.apply_along_axis(self._fill_inf_with_zero, axis=0, arr=data_)
        
        return data_filled

    def _fill_inf_with_zero(self, arr):
        '''
        fill inf 调用的内部函数
        '''
        a = [0 if math.isinf(x) else x for x in arr ]
        return np.array(a)

    def _fill_inf_with_peak(self, arr):
        '''
        fill inf 调用的内部函数
        '''
        a = arr
        arrmax = 1e1
        arrmin = 1e-1
        posinf = np.isposinf(a)
        neginf = np.isneginf(a)
        a[posinf] = arrmax
        a[neginf] = arrmin
        
        return a

    @info
    def convert_log(self, dataframe, trigger=10):
        '''
        对数值超过触发门限的列 取对数，对负数取绝对值再取对数，结果再取负
        '''
        # 根据门限找出需要取对数的数据列
        data_ = dataframe.values
        col_max = np.apply_along_axis(np.max, axis=0, arr=data_)
        col_min = np.apply_along_axis(np.min, axis=0, arr=data_)
        log_col_1 = dataframe.columns.values[np.where(col_max >= trigger)]
        log_col_2 = dataframe.columns.values[np.where(col_min <= -trigger)]
        log_col = np.concatenate([log_col_1, log_col_2], axis=0)
        log_col = np.unique(log_col)
        no_log_col = np.array([x for x in dataframe.columns.values if x not in log_col])
        
        # 计算两个区分的数据形状，便于合并
        no_log_data = dataframe[no_log_col].values
        no_log_shape = dataframe[no_log_col].values.shape
        log_data = dataframe[log_col].values
        log_shape = dataframe[log_col].values.shape

        log_neg = np.where(log_data < 0)
        log_pos = np.where(log_data > 0)
        log_zero = np.where(log_data == 0)

        log_ = np.zeros(shape=log_data.shape)
        log_[log_pos] = np.log10(log_data[log_pos])
        log_[log_neg] = -np.log10(np.abs(log_data[log_neg]))


        if len(log_shape) > len(no_log_shape):
            no_log_shape.append(1)
        elif len(log_shape) < len(no_log_shape):
            log_shape.append(1)

        # 对已经计算和未计算的数据列进行拼接
        res = np.concatenate([dataframe[no_log_col].values.reshape(no_log_shape), log_.reshape(log_shape)], axis=1)

        return res

    @info
    def cal_technical_indicators(self, data, plot=False, save=False):
        '''
        计算股价技术指标 
        输入参数为数据集、持续时间和是否绘制图表 输出技术指标 key表示对哪一个指标进行统计分析
        7日均线和21日均线
        '''
        import stockstats

        last_days = data.shape[0]

        dataset_tech = data[['daily_open', 'daily_close', 'daily_high', 'daily_low', 'daily_vol', 'daily_amount']]
        dataset_tech = dataset_tech.rename(columns= lambda x: x.lstrip('daily_')).rename(columns={'vol':'volume', 'ow':'low', 'mount':'amount'})
        stock = stockstats.StockDataFrame(dataset_tech)
        technical_keys = ['macd', # moving average convergence divergence. Including signal and histogram. 
                            'macds',# MACD signal line
                            'macdh', # MACD histogram

                            'volume_delta', # volume delta against previous day
                            'volume_-3,2,-1_max', # volume max of three days ago, yesterday and two days later
                            'volume_-3~1_min', # volume min between 3 days ago and tomorrow

                            'kdjk', # KDJ, default to 9 days
                            'kdjd', 
                            'kdjj',
                            'kdjk_3_xu_kdjd_3', # three days KDJK cross up 3 days KDJD

                            'boll', # bolling, including upper band and lower band
                            'boll_ub',
                            'boll_lb', 

                            'open_2_sma', # 2 days simple moving average on open price
                            'open_2_d', # open delta against next 2 day
                            'open_-2_r', # open price change (in percent) between today and the day before yesterday, 'r' stands for rate.
                            'close_10.0_le_5_c', # close price less than 10.0 in 5 days count

                            'cr', # CR indicator, including 5, 10, 20 days moving average
                            'cr-ma1', 
                            'cr-ma2', 
                            'cr-ma3', 
                            'cr-ma2_xu_cr-ma1_20_c', # CR MA2 cross up CR MA1 in 20 days count

                            'rsi_6', # 6 days RSI
                            'rsi_12', # 12 days RSI

                            'wr_10', # 10 days WR
                            'wr_6', # 6 days WR

                            'cci', # CCI, default to 14 days
                            'cci_20', ## 20 days CCI

                            'dma', # DMA, difference of 10 and 50 moving average
                            'pdi', # DMI  +DI, default to 14 days
                            'mdi', # -DI, default to 14 days
                            'dx', # DX, default to 14 days of +DI and -DI
                            'adx', # ADX, 6 days SMA of DX, same as stock['dx_6_ema']
                            'adxr', # ADXR, 6 days SMA of ADX, same as stock['adx_6_ema']

                            'tr', #TR (true range)
                            'atr', # ATR (Average True Range)
                            'trix', # TRIX, default to 12 days
                            'trix_9_sma', # MATRIX is the simple moving average of TRIX

                            'vr', # VR, default to 26 days
                            'vr_6_sma' # MAVR is the simple moving average of VR
                            ]
        for key in technical_keys:
            dataset_tech[key] = pd.DataFrame(stock[key])

        dataset_tech['ma7'] = dataset_tech['close'].rolling(window=7).mean()
        dataset_tech['ma21'] = dataset_tech['close'].rolling(window=21).mean()
        dataset_tech['ema'] = dataset_tech['close'].ewm(com=0.5).mean()
        dataset_tech['momentum'] = dataset_tech['close']-1

        if plot:# 绘制技术指标
            plot_dataset = dataset_tech
            plot_dataset = dataset_tech.iloc[-last_days:, :]
            shape_0 = plot_dataset.shape[0]
            x = list(plot_dataset.index)
            colors = self._choose_color(10)

            # 0.股价、成交量，成交额、MA移动平均线
            plt.figure(figsize=(16,10), dpi=150)
            linewidth = 1
            plt.subplot(3, 1, 1)
            plt.title('Close Price and Volume Statistics')
            plt.plot(plot_dataset['close'], label='Close Price')
            plt.plot(plot_dataset['ma7'], label='MA-7', linestyle='--', linewidth=linewidth)
            plt.plot(plot_dataset['ma21'], label='MA-21', linestyle='--', linewidth=linewidth)
            plt.plot(plot_dataset['ema'], label='EMA', linestyle=':', linewidth=linewidth)
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.bar(x, plot_dataset['volume'], label='Volume', width=linewidth)
            plt.bar(x, -plot_dataset['amount'], label='Amount', width=linewidth )
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(plot_dataset['volume_delta'], label='volume delta', linestyle='-', linewidth=linewidth/2, color='k')
            plt.legend()
            if save:
                plt.savefig('project\\feature_engineering\\30_price_amount.png')
            plt.show()

            # 1.MACD 
            plt.figure(figsize=(16,10), dpi=150)
            linewidth = 1
            plt.subplot(2, 1, 1)
            plt.title('Close price and MACD')
            plt.plot(plot_dataset['close'], label='Close Price')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(plot_dataset['macd'], label='macd', linewidth=linewidth)
            plt.plot(plot_dataset['macds'], label='macd signal line', linestyle='--', linewidth=linewidth)
            plt.bar(plot_dataset['macdh'].loc[plot_dataset['macdh']>=0].index, plot_dataset['macdh'].loc[plot_dataset['macdh']>=0], label='macd histgram', width=linewidth, color='r')
            plt.bar(plot_dataset['macdh'].loc[plot_dataset['macdh']<0].index, plot_dataset['macdh'].loc[plot_dataset['macdh']<0], label='macd histgram', width=linewidth, color='g')
            plt.legend()
            if save:
                plt.savefig('project\\feature_engineering\\31_MACD.png')
            plt.show()

            # 2.KDJ and BOLL
            plt.figure(figsize=(16,10), dpi=150)
            linewidth = 1
            plt.subplot(2, 1, 1)
            plt.title('Bolling band and KDJ')
            plt.plot(plot_dataset['close'], label='Close Price')
            plt.plot(plot_dataset['boll'], label='Bolling', linestyle='--', linewidth=linewidth)
            plt.plot(plot_dataset['boll_ub'],color='c', label='Bolling up band', linewidth=linewidth)
            plt.plot(plot_dataset['boll_lb'],color='c', label='Bolling low band', linewidth=linewidth)
            plt.fill_between(x, plot_dataset['boll_ub'], plot_dataset['boll_lb'], alpha=0.35)
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(plot_dataset['kdjk'], label='KDJ-K', linewidth=linewidth)
            plt.plot(plot_dataset['kdjd'], label='KDJ-K', linewidth=linewidth)
            plt.plot(plot_dataset['kdjj'], label='KDJ-K', linewidth=linewidth)
            plt.scatter(plot_dataset['kdjk'].loc[plot_dataset['kdjk_3_xu_kdjd_3']==True].index, plot_dataset['kdjk'].loc[plot_dataset['kdjk_3_xu_kdjd_3']==True], 
                        marker='^', color='r', label='three days KDJK cross up 3 days KDJD')
            plt.legend()
            if save:
                plt.savefig('project\\feature_engineering\\32_boll_kdj.png')
            plt.show()

            # 3.Open price and RSI
            plt.figure(figsize=(16,10), dpi=150)
            linewidth = 1
            plt.subplot(2, 1, 1)
            plt.title('Open price and RSI')
            plt.plot(plot_dataset['open'], label='Open Price')
            plt.bar(x, plot_dataset['open_2_d'], label='open delta against next 2 day')
            plt.plot(plot_dataset['open_-2_r'], label='open price change (in percent) between today and the day before yesterday', linewidth=linewidth)
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(plot_dataset['rsi_12'], label='12 days RSI ', color='c') 
            plt.plot(plot_dataset['rsi_6'], label='6 days RSI', linewidth=linewidth, color='r')  
            plt.legend()
            if save:
                plt.savefig('project\\feature_engineering\\33_open_rsi.png')
            plt.show()

            # 4.CR and WR

            plt.figure(figsize=(16,10), dpi=150)
            linewidth = 1
            plt.subplot(2, 1, 1)
            plt.title('WR and CR in 5/10/20 days')
            plt.plot(plot_dataset['wr_10'], label='10 days WR', linestyle='-', linewidth=linewidth, color='g')
            plt.plot(plot_dataset['wr_6'], label='6 days WR', linestyle='-', linewidth=linewidth, color='r')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.bar(x, plot_dataset['cr'], label='CR indicator', linestyle='--', linewidth=linewidth, color='skyblue')
            plt.plot(plot_dataset['cr-ma1'], label='CR 5 days MA', linestyle='-', linewidth=linewidth)
            plt.plot(plot_dataset['cr-ma2'], label='CR 10 days MA', linestyle='-', linewidth=linewidth)
            plt.plot(plot_dataset['cr-ma3'], label='CR 20 days MA', linestyle='-', linewidth=linewidth)
            plt.legend()
            if save:
                plt.savefig('project\\feature_engineering\\34_cr_ma.png')
            plt.show()

            # 5.CCI TR VR 
            # 
            plt.figure(figsize=(16,10), dpi=150)
            linewidth = 1
            plt.subplot(2, 1, 1)
            plt.title('CCI TR and VR')
            plt.plot(plot_dataset['tr'], label='TR (true range)', linewidth=linewidth)
            plt.plot(plot_dataset['atr'], label='ATR (Average True Range)', linewidth=linewidth)
            plt.plot(plot_dataset['trix'], label='TRIX, default to 12 days', linewidth=linewidth)
            plt.plot(plot_dataset['trix_9_sma'], label='MATRIX is the simple moving average of TRIX', linewidth=linewidth)
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(plot_dataset['cci'], label='CCI, default to 14 days', linestyle='-', linewidth=linewidth, color='r')
            plt.plot(plot_dataset['cci_20'], label='20 days CCI', linestyle='-', linewidth=linewidth, color='g')
            plt.bar(x, plot_dataset['vr'], label='VR, default to 26 days')
            plt.bar(x, -plot_dataset['vr_6_sma'], label='MAVR is the simple moving average of VR')
            plt.legend()
            if save:
                plt.savefig('project\\feature_engineering\\35_cci_tr_vr.png')
            plt.show()

            # 6.DMI
            plt.figure(figsize=(16,10), dpi=150)
            linewidth = 1
            plt.subplot(3, 1, 1)
            plt.title('DMI and DMA')
            plt.bar(x, plot_dataset['pdi'], label='+DI, default to 14 days', color='r')
            plt.bar(x, -plot_dataset['mdi'], label='-DI, default to 14 days', color='g')
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(plot_dataset['dma'], label='DMA, difference of 10 and 50 moving average', linewidth=linewidth, color='k')
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(plot_dataset['dx'], label='DX, default to 14 days of +DI and -DI', linewidth=linewidth)
            plt.plot(plot_dataset['adx'], label='6 days SMA of DX', linewidth=linewidth)
            plt.plot(plot_dataset['adxr'], label='ADXR, 6 days SMA of ADX', linewidth=linewidth)
            plt.legend()
            if save:
                plt.savefig('project\\feature_engineering\\36_close_DMI.png')
            plt.show()

        return dataset_tech   

    @info
    def cal_fft(self, data, plot=False, save=False):
        '''
        计算傅里叶变换 

        输入
            股价数据集 
            
        输出
            傅里叶变换
        
        '''
        from scipy.fftpack import fft, ifft

        data_FT = data['daily_close'].astype(float)
        technical_data = np.array(data_FT, dtype=float)
        close_fft = fft(technical_data)
        fft_df = pd.DataFrame({'fft_real':close_fft.real, 'fft_imag':close_fft.imag, 'fft':close_fft})
        fft_df['fft_absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        fft_df['fft_angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

        if plot:# 绘制傅里叶变换的图像
            plt.figure(figsize=(14, 7), dpi=100)
            plt.plot(data_FT, label='Close Price')
            fft_list = np.array(fft_df['fft'])
            for num_ in [3, 9, 27, 100]:
                fft_list_m10 = np.copy(fft_list)
                fft_list_m10[num_:-num_]=0
                ifft_list = pd.DataFrame(ifft(fft_list_m10)).set_index(data_FT.index)
                plt.plot(ifft_list, label='Fourier transform with {} components'.format(num_))
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.title('Stock prices & Fourier transforms')
            plt.legend()
            if save:
                plt.savefig('project\\feature_engineering\\40_Fourier_transforms.png')
            plt.show()

            from collections import deque
            items = deque(np.asarray(fft_df['fft_absolute'].tolist()))
            items.rotate(int(np.floor(len(fft_df)/2)))
            # 绘制的频谱数量
            plot_len = 100
            items_plot = list(items)
            items_plot = items_plot[int(len(fft_df)/2-plot_len/2) : int(len(fft_df)/2+plot_len/2)]

            plt.figure(figsize=(10, 7), dpi=100)
            plt.stem(items_plot)
            plt.title(str(plot_len) + ' Components of Fourier transforms ')
            if save:
                plt.savefig('project\\feature_engineering\\41_Fourier_components.png')
            plt.show()

        fft_ = fft_df.set_index(data_FT.index).drop(columns='fft') # 去掉复数的部分

        return fft_

    @info
    def encode_date_embeddings(self, timeseries=None):
        '''
        对时间进行编码

            周期：[10年, 季度of年，月of年，天of周，天of月]
        '''
        T = [10, 4, 12, 7, 0]
        PI = math.pi

        if isinstance(timeseries, np.ndarray):
            date_list = []
            ts = [(x-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's') for x in timeseries]
            for x in ts:
                try:
                    tmp = arrow.get(x)
                    date_list.append(tmp)
                except Exception as e:
                    print(e)
        elif isinstance(timeseries, pd.Series):
            date_list = pd.to_datetime(timeseries, format='%Y%m%d').to_list()
            date_list = [arrow.get(t) for t in date_list]
        else:
            date_list = [arrow.get(timeseries)]

        embedding_list = []
        for dt in date_list:
            T[4] = calendar.monthrange(dt.year, dt.month)[1]
            d_of_w = calendar.weekday(dt.year, dt.month, dt.day)
            q_of_y = math.ceil(dt.month/3)
            datetime_vec = [dt.year, q_of_y, dt.month,
                            d_of_w, dt.day]
            x = np.array(datetime_vec) % np.array(T)
            x_t = x/T
            sin_ = [math.sin(PI*i) for i in x_t]
            cos_ = [math.cos(PI*i) for i in x_t]
            embedding = sin_ + cos_
            embedding_list.append(embedding)

        # 转换为字符串
        date_list = [x.strftime('%Y%m%d')
                           for x in date_list]
        assert len(date_list) == len(embedding_list)

        return np.array(date_list), np.array(embedding_list)

    @info
    def decode_date_embeddings(self, embeddings=None):
        '''
        对于已经编码的向量进行解码，解码成时间字符串

        周期：[10年, 季度of年，月of年，天of周，天of月]
        '''
        T = [10, 4, 12, 7, 0]
        start_year = 2009
        PI = math.pi

        assert len(embeddings) == 10
        arcsin_ = [math.asin(x)/(PI) for x in embeddings[:5]]
        arccos_ = [math.acos(x)/(PI) for x in embeddings[-5:]]
        arc_T = np.array((arcsin_+arccos_)) * np.array((T+T))

        pass

    @info
    def cal_daily_quotes(self, data):
        '''
        计算每日行情，处理价和量
        '''
        daily_quotes = self.data_cfg['daily_quotes']
        target_col = self.data_cfg['target']
        price_col = daily_quotes[1:6]
        price_col.pop(3)
        volumn_col = daily_quotes[-2:]
        change_col = daily_quotes[6:8]
        price_ = data[price_col]
        volumn_ = data[volumn_col]
        target_ = data[target_col]
        change_ = data[change_col]

        # 计算开收高低与收盘价的差价的百分比
        diff_price_pct = (price_.values - target_.values.reshape((-1, 1))) * 100 / target_.values.reshape((-1, 1))
        # 计算成交量和成交额的对数
        log_vol = np.log10(volumn_.values)

        daily_quote_feature = np.concatenate([target_.values.reshape((-1, 1)), diff_price_pct, log_vol], axis=1)

        return daily_quote_feature


    @info
    def concat_features(self, datalist, pca_comp=0.99):
        '''
        将所有除了每日行情之外的特征进行拼接，并且完成去重、消除空值、无穷值、将数量级较大的数据缩放取对数
        '''
        from sklearn.decomposition import PCA

        full_data = []
        for data in datalist:
            data_ = self.drop_dup_fill_nan(pd.DataFrame(data))
            data_ = self.fill_inf(pd.DataFrame(data_))
            data_ = self.convert_log(pd.DataFrame(data_))
            full_data.append(data_)

        full_data_ = np.concatenate(full_data, axis=1)

        pca = PCA(n_components=pca_comp)

        pca_data = pca.fit_transform(full_data_)

        return pca_data


class DataVisualiser():
    '''
    数据可视化器
    '''
    def __init__(self, config):
        self.config = config

    @info
    def get_ARIMA(self, data, plot=True, save=False):# 获取时间序列特征，使用ARIMA 输入为股价数据集
        from statsmodels.tsa.arima_model import ARIMA
        from pandas import DataFrame
        from pandas import datetime
        from pandas import read_csv
        from pandas import datetime
        from statsmodels.tsa.arima_model import ARIMA
        from sklearn.metrics import mean_squared_error

        series = data['daily_close'].astype(float)
        model = ARIMA(series, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        # print(model_fit.summary())
        summary = model_fit.summary()

        if plot:
            from pandas.plotting import autocorrelation_plot
            plt.figure()
            autocorrelation_plot(series, label='Close price correlations')
            if save:
                plt.savefig('project\\feature_engineering\\50_Close_price_correlations.png')

        X = series.values
        size = int(len(X) * 0.9)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)

        error = mean_squared_error(test, predictions)
        print('Test MSE: %.3f' % error)

        if plot:
            plt.figure(figsize=(12, 6), dpi=100)
            plt.plot(test, label='Real')
            plt.plot(predictions, color='red', label='Predicted')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.title('ARIMA model')
            plt.legend()
            if save:
                plt.savefig('project\\feature_engineering\\51_ARIMA_model.png')
            plt.show()
        
        return summary, predictions

    @info
    def get_feather_importance(self, data, plot=True, save=False):# 获取特征重要性指标 通过xgboost验证
        import xgboost as xgb

        y = data['daily_close'].astype(float)
        # 训练数据中的特征，因为开盘价、收盘价、最高价、最低价都与收盘价y强相关，这些特征会影响其他特征的作用
        # 所以在评估时，将其删除
        # 以下是在测试中重要性大于0.2的特征
        X = data.drop(columns=['cal_date','daily_close','daily_open','daily_low','daily_high','tech_momentum',
                                'tech_ma7', 'tech_ma21', 'tech_ema', 'tech_middle', 'tech_close_-1_s', 'tech_open_2_sma', 'tech_open_2_s',
                                'tech_boll_lb', 'tech_close_10_sma', 'tech_close_10.0_le', 'tech_middle_14_sma',
                                'tech_middle_20_sma', 'tech_close_20_sma', 'tech_close_26_ema','tech_boll','tech_boll_ub',
                                'daily_pre_close','res_qfq_close','res_hfq_close','tech_close_50_sma','tech_atr_14',
                                'tech_atr'
                                ])

        train_samples = int(X.shape[0] * 0.9)
        X_train = X.iloc[:train_samples]
        X_test = X.iloc[train_samples:]
        y_train = y.iloc[:train_samples]
        y_test = y.iloc[train_samples:]

        regressor = xgb.XGBRegressor(gamma=0.0,
                                    n_estimators=150,
                                    base_score=0.7,
                                    colsample_bytree=1,
                                    learning_rate=0.05,
                                    objective='reg:squarederror')
        xgbModel = regressor.fit(X_train,y_train,
                             eval_set = [(X_train, y_train), (X_test, y_test)],
                             verbose=False)
        eval_result = regressor.evals_result()
        training_rounds = range(len(eval_result['validation_0']['rmse']))
        importance = xgbModel.feature_importances_.tolist()
        feature = X_train.columns.tolist()
        feature_importance = pd.DataFrame({'Importance':importance}, index=feature)

        plot_importance = feature_importance.nlargest(40, columns='Importance')
        # 取前40个最重要的特征

        if plot:
            plt.plot(training_rounds,eval_result['validation_0']['rmse'],label='Training Error')
            plt.plot(training_rounds,eval_result['validation_1']['rmse'],label='Validation Error')
            plt.xlabel('Iterations')
            plt.ylabel('RMSE')
            plt.title('Training Vs Validation Error')
            plt.legend()
            if save:
                plt.savefig('project\\feature_engineering\\60_Training_Vs_Validation_Error.png')
            plt.show()

            fig = plt.figure(figsize=(16,8))
            plt.barh(plot_importance.index, plot_importance['Importance'])
            plt.title('Feature importance of the data')
            if save:
                plt.savefig('project\\feature_engineering\\61_Feature_importance.png')
            plt.show()

        return feature_importance
