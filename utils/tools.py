import time
import datetime as dt
import sys,os
import logging
from functools import wraps
import traceback
import arrow

class Timer():
    '''
    定义一个计时器类 stop方法输出用时
    '''

    def __init__(self, name=None):
        self.start_dt = None
        self.name = name

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        
        print('[Timer \"%s\"] Time taken: %s' % (self.name, end_dt - self.start_dt))

class Logger():
    '''
    一个日志记录器
    '''
    def __init__(self,):
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),"logs")
        today = time.strftime('%Y%m%d', time.localtime(time.time()))
        full_path = os.path.join(log_dir, today)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        self.log_path = os.path.join(full_path,"quant.log")

    def get_logger(self, ):
     # 获取logger实例，如果参数为空则返回root logger
        self.logger = logging.getLogger("quant")
        if not self.logger.handlers:
            # 指定logger输出格式
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
 
            # 文件日志
            file_handler = logging.FileHandler(self.log_path, encoding="utf8")
            file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
 
            # 控制台日志
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter  # 也可以直接给formatter赋值
 
            # 为logger添加的日志处理器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
 
            # 指定日志的最低输出级别，默认为WARN级别
            self.logger.setLevel(logging.INFO)
     #  添加下面一句，在记录日志之后移除句柄
        return  self.logger


def info(func):
    @wraps(func)
    def log(*args,**kwargs):
        logger = Logger()
        try:
            logger.get_logger().info("Method: \" {name} \" is starting...".format(name = func.__name__))
            timer = Timer(func.__name__)
            timer.start()
            result = func(*args,**kwargs)
            timer.stop()
            logger.get_logger().info("Method: \" {name} \" is completed .".format(name = func.__name__))
            return result
        except Exception as e:
            logger.get_logger().error(f"{func.__name__} is error,here are details:{traceback.format_exc()}")
    return log



def search_file(path=None, filename=None):
    """
    递归查询文件下包含指定字符串的文件
    """
    res = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            search_file(item_path, filename)
        elif os.path.isfile(item_path):
            if filename in item_path:
                res.append(item_path)
    return res


def parse_filename(filename:str):
    """
    解析文件名，并返回一个字典
    """
    f = filename.strip()
    if f.endswith('.h5'):
        f = f[:-3]
        args = f.split('-')
        assert len(args) == 7
        args_dict = {
            'train_date': arrow.get(args[0][-15:], 'YYYYMMDD_HHmmss'),
            'loss': float(args[1]),
            'val_loss': float(args[2]),
            'acc': float(args[3]),
            'val_acc': float(args[4]),
            'stock': args[5],
            'end_date':arrow.get(args[6], 'YYYYMMDD')
        }
        return args_dict
    else:
        return None


