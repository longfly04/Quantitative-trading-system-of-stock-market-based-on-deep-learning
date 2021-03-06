import os
import json
import types
import datetime
import random
import matplotlib.dates as mdates
import numpy as np

import numpy as np
from gym.core import GoalEnv
from gym import spaces

import logging


logger = logging.getLogger('trading-gym')


class TradeEnv(GoalEnv, ExtraFeature):

    def __init__(self,
                 data=None,
                 data_path=None,
                 data_func=None,
                 previous_steps=60,
                 max_episode=200,
                 punished=False,
                 nav=5000,
                 get_obs_features_func=None,
                 ops_shape=None,
                 get_reward_func=None,
                 add_extra=False,
                 data_kwargs={}):
        self.data = DataManager(
            data, data_path, data_func, previous_steps,
            **data_kwargs)
        self.exchange = Exchange(punished=punished, nav=nav)
        self._render = Render()

        self.get_obs_features_func = get_obs_features_func
        self.get_reward_func = get_reward_func
        self.add_extra = add_extra

        if self.get_obs_features_func is not None:
            if ops_shape is None:
                raise ValueError(
                    "ops_shape should be given if use get_obs_features_func")
            self.ops_shape = ops_shape
        elif self.data.use_ta:
            self.ops_shape = self.data.ta_data.feature_space
        else:
            self.ops_shape = self.data.default_space

        if len(self.ops_shape) == 2 and self.add_extra:
            self.ops_shape[-1] = self.ops_shape[-1] + len(self.ex_obs_name)

        self.obs = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=self.ops_shape,
                                            dtype=np.float32)

    def get_obs(self, info):
        history = self.data.recent_history

        if self.get_obs_features_func is not None:
            obs = self.get_obs_features_func(history, info)
        elif self.data.use_ta:
            obs = self.data.ta_features
        else:
            obs = self.data.recent_history.to_array(
                base=self.data.first_price,
                extend=[info['avg_price']])

        if self.add_extra:
            extra_obs = self.get_extra_features(info)
            if len(obs.shape) == 2:
                extra_obs = np.ones([obs.shape[0], len(extra_obs)]) * extra_obs
                obs = np.concatenate((obs, extra_obs), axis=1)
            else:
                obs = np.concatenate((obs, extra_obs))
        return obs

    def get_reward(self, profit):
        if self.get_reward_func is not None:
            return self.get_reward_func(self.exchange)
        else:
            return profit

    def _step(self, action):
        if self.obs is None:
            self.obs, done = self.data.step()

        if action in self.exchange.available_actions:
            self._render.take_action(action, self.obs)

        profit = self.exchange.step(action, self.obs)
        reward = self.get_reward(profit)

        info = self.exchange.info

        obs = self.get_obs(info)
        self.obs, done = self.data.step()

        if self.exchange.is_over_loss:
            done = True

        return obs, reward, done, info

    def step(self, action):
        action = action - 1
        observation, reward, done, info = self._step(action)
        return observation, reward, done, info

    def reset(self, index=None):
        self.data.reset(index)
        self.exchange.reset()
        self._render.reset()
        observation, _, _, _ = self._step(
            self.exchange.start_action)
        return observation

    def render(self, mode='human', close=False):
        history = self.data.history
        info = self.exchange.info
        self._render.render(history, info, mode=mode, close=close)

    def close(self):
        return


class ACTION:
    PUT = -1
    HOLD = 0
    PUSH = 1


class Positions:

    def __init__(self, symbol=''):
        self.symbol = symbol
        self.amount = 0
        self.avg_price = 0
        self.created_index = 0

    @property
    def principal(self):
        return self.avg_price * self.amount

    @property
    def is_empty(self):
        return self.amount == 0

    @property
    def is_do_long(self):
        return self.amount > 0

    @property
    def is_do_short(self):
        return self.amount < 0

    @property
    def rate(self):
        if self.is_do_short:
            diff = abs(self.avg_price) - self.latest_price
        else:
            diff = self.latest_price - abs(self.avg_price)
        return diff / self.avg_price if self.avg_price else 0

    def get_profit(self, latest_price, nav):
        if self.is_empty:
            return 0
        else:
            profit = self.rate * nav
            return profit

    def update(self, action, latest_price):
        self.latest_price = latest_price

    def step(self, action, nav, index):
        if action != 0 and self.amount == 0:
            self.created_index = index
        if action * self.amount < 0:
            profit = self.get_profit(self.latest_price, nav)
        else:
            profit = 0

        amount = action * nav
        if self.amount + amount != 0:
            total = self.avg_price * self.amount + amount * self.latest_price
            self.avg_price = total / (self.amount + amount)
        else:
            self.avg_price = 0
            self.created_index = 0
        self.amount += amount
        return profit


class Transaction(object):
    _columns = ["index", "time", "latest_price", "action", "amount", "profit",
                "fixed_profit", "floating_profit"]

    def init__transaction(self):
        self.transaction = pd.DataFrame(columns=self._columns)

    def add_transaction(self, obs, action):
        current = pd.DataFrame({
            'index': [obs.index],
            'time': [obs.date_string],
            'latest_price': [obs.latest_price],
            'action': [action],
            'amount': [self.amount],
            'profit': [self.profit],
            'fixed_profit': [self.fixed_profit],
            'floating_profit': [self.floating_profit]
        })
        self.transaction = pd.concat(
            [self.transaction, current], ignore_index=True)


class Exchange(Transaction):

    """

    Attributes:
        fixed_profit (int): fixed_profit
        floating_profit (int): floating_profit
        nav (int): nav
        observation (Observation): observation
        position (Positions): position
        punished (bool): punished
        unit (int): unit
    """

    def __init__(self, punished=False, nav=50000, end_loss=None, unit=5000):
        """

        Args:
            punished (bool, optional): Do punishe if True
            nav (int, optional): nav
            end_loss (None, optional): end_loss
            unit (int, optional): unit
        """
        self.nav = nav
        self.punished = punished
        self._end_loss = end_loss
        self.unit = unit
        self.__init_data()

    def __init_data(self):
        self.position = Positions()
        self.fixed_profit = 0
        self.init__transaction()

    @property
    def start_action(self):
        return ACTION.HOLD

    @property
    def available_funds(self):
        return self.nav - self.position.principal

    @property
    def floating_rate(self):
        return self.position.rate

    @property
    def profit(self):
        return self.fixed_profit + self.floating_profit

    @property
    def available_actions(self):
        if self.position.is_do_short:
            return [ACTION.PUSH, ACTION.HOLD]
        elif self.position.is_do_long:
            return [ACTION.PUT, ACTION.HOLD]
        else:
            return [ACTION.PUT, ACTION.HOLD, ACTION.PUSH]

    @property
    def punished_action(self):
        if self.position.is_do_short:
            return ACTION.PUT
        elif self.position.is_do_long:
            return ACTION.PUSH
        return None

    @property
    def cost_action(self):
        return [ACTION.PUT, ACTION.PUSH]

    @property
    def end_loss(self):
        if self._end_loss is not None:
            return self._end_loss
        return - self.nav * 0.2

    @property
    def is_over_loss(self):
        return self.profit < self.end_loss

    @property
    def amount(self):
        return self.position.amount / self.nav

    @property
    def floating_profit(self):
        return self.position.get_profit(self.latest_price, self.nav)

    # def get_profit(self, observation, symbol='default'):
    #     latest_price = observation.latest_price
    #     positions = self.positions.values()
    #     positions_profite = sum([position.get_profit(
    #         latest_price) for position in positions])
    #     return self.profit + positions_profite

    def get_charge(self, action, latest_price):
        """
            rewrite if inneed.
        """
        if self.position.is_empty and action in self.cost_action:
            amount = self.nav / latest_price
            if amount < 100:
                return 2
            else:
                return amount * (0.0039 + 0.0039)
        else:
            return 0

    def step(self, action, observation, symbol='default'):
        self.observation = observation
        self.latest_price = observation.latest_price
        self.position.update(action, self.latest_price)

        charge = self.get_charge(action, self.latest_price)
        if action in self.available_actions:
            fixed_profit = self.position.step(
                action, self.nav, observation.index)
            fixed_profit -= charge
            self.fixed_profit += fixed_profit

            self.add_transaction(observation, action)
        else:
            fixed_profit = 0
        # if self.punished:
        #     if action == self.punished_action:
        #         fixed_profit -= 2  # make it different
        #     if action == 0:
        #         fixed_profit -= 2  # make it action

        logger.info(f"latest_price:{observation.latest_price}, "
                    f"amount:{self.position.amount}, "
                    f"action:{action}")
        logger.info(f"fixed_profit: {self.fixed_profit}, "
                    f"floating_profit: {self.floating_profit}")
        return self.fixed_profit / self.nav

    def reset(self):
        self.__init_data()

    @property
    def info(self):
        return {
            'index': self.observation.index,
            'date': self.observation.date_string,
            'nav': self.nav,
            'amount': self.position.amount,
            'avg_price': self.position.avg_price,
            'profit': {
                'total': self.profit,
                'fixed': self.fixed_profit,
                'floating': self.floating_profit
            },
            'buy_at': self.position.created_index,
            'latest_price': self.observation.latest_price,
        }


class Arrow(object):

    def __init__(self, coord, color):
        self.coord = coord
        self.color = color


class Render(object):

    def __init__(self, bar_width=0.8):
        plt.ion()
        self.figure, self.ax = plt.subplots()
        self.figure.set_size_inches((12, 6))
        self.bar_width = 0.8
        self.arrow_body_len = self.bar_width / 1000
        self.arrow_head_len = self.bar_width / 20
        self.arrow_width = self.bar_width
        self.max_windows = 60

        self.arrows = []

    @property
    def arrow_len(self):
        return self.arrow_body_len + self.arrow_head_len

    def take_action(self, action, observation):
        if action == ACTION.PUT:
            y_s = observation.high + self.arrow_body_len + self.arrow_head_len
            y_e = - self.arrow_body_len
            color = 'gray'
        elif action == ACTION.PUSH:
            y_s = observation.high
            y_e = self.arrow_body_len
            color = 'black'
        else:
            return
        x = observation.index
        self.arrows.append(Arrow((x, y_s, 0, y_e), color))

    def draw_arrow(self, data):
        first = data[0]
        arrows = list(filter(lambda x: x.coord[0] > first[0], self.arrows))
        for arrow in arrows:
            self.ax.arrow(*arrow.coord,
                          color=arrow.color,
                          head_width=self.arrow_width,
                          head_length=self.arrow_head_len)

        self.arrows = arrows

    def draw_title(self, info):
        formatting = """
        profit: {} \t fixed-profit: {} \t floating-profit:{} \t
        amount: {} \t latest-price: {} \t time: {} \n
        """

        def r(x): return round(x, 2)
        title = formatting.format(r(info['profit']['total']),
                                  r(info['profit']['fixed']),
                                  r(info['profit']['floating']),
                                  info['amount'],
                                  r(info['latest_price']),
                                  info['date'])
        plt.suptitle(title)

    def xaxis_format(self, history):
        def formator(x, pos=None):
            for h in history:
                if h.index == x:
                    return h.date.strftime("%H:%M")
            return ""
        return formator

    def render(self, history, info, mode='human', close=False):
        """
            If it plot one by one will cache many point
            that will cause OOM and work slow.
        """
        self.ax.clear()
        history = history[-self.max_windows:]
        data = [obs.to_list() for obs in history]
        candlestick_ochl(self.ax, data, colordown='g',
                         colorup='r', width=self.bar_width)
        self.draw_arrow(data)

        self.ax.xaxis.set_major_formatter(
            FuncFormatter(self.xaxis_format(history)))
        self.ax.set_ylabel('price')
        self.figure.autofmt_xdate()

        self.draw_title(info)
        plt.show()
        plt.pause(0.0001)

    def reset(self):
        self.ax.clear()
        self.arrows = []


class Observation(object):
    __slots__ = ["open", "close", "high", "low", "index", "volume",
                 "date", "date_string", "format_string"]
    N = 6

    def __init__(self, **kwargs):
        self.index = kwargs.get('index', 0)
        self.open = kwargs.get('open', 0)
        self.close = kwargs.get('close', 0)
        self.high = kwargs.get('high', 0)
        self.low = kwargs.get('low', 0)
        self.volume = kwargs.get('volume', 0)
        self.date_string = kwargs.get('date', '')
        self.date = self._format_time(self.date_string)

    @property
    def math_date(self):
        return mdates.date2num(self.date)

    @property
    def math_hour(self):
        return (self.date.hour * 100 + self.date.minute) / 2500

    def _format_time(self, date):
        d_f = date.count('-')
        t_f = date.count(':')
        if d_f == 2:
            date_format = "%Y-%m-%d"
        elif d_f == 1:
            date_format = "%m-%d"
        else:
            date_format = ''

        if t_f == 2:
            time_format = '%H:%M:%S'
        elif t_f == 1:
            time_format = '%H:%M'
        else:
            time_format = ''
        self.format_string = " ".join([date_format, time_format]).rstrip()
        return datetime.datetime.strptime(date, self.format_string)

    def to_ochl(self):
        return [self.open, self.close, self.high, self.low]

    def to_list(self):
        return [self.index] + self.to_ochl()

    def to_array(self, extend):
        volume = self.volume / 10000
        extend = [volume] + extend
        return np.array(self.to_ochl() + extend, dtype=float)

    def __str__(self):
        return "date: {self.date}, open: {self.open}, close:{self.close}," \
            " high: {self.high}, low: {self.low}".format(self=self)

    @property
    def latest_price(self):
        return self.close


class History(object):

    def __init__(self, obs_list, history_num):
        self.obs_list = obs_list
        self.history_num = history_num

    def normalize(self, base):
        def __nor(array):
            return [(x - base) / base for x in array]
        return __nor

    def to_array(self, base, extend=[]):
        # base = self.obs_list[-1].close if self.obs_list else 1

        history = np.zeros([self.history_num + 1, Observation.N], dtype=float)
        normalize_func = self.normalize(base)
        for i, obs in enumerate(self.obs_list):
            data = obs.to_array(extend=extend)
            history[i] = normalize_func(data)
        return history


class DataManager(object):

    def __init__(self, data=None,
                 data_path=None,
                 data_func=None,
                 previous_steps=None,
                 history_num=50,
                 start_random=False,
                 use_ta=False,
                 ta_timeperiods=None,
                 ta_class=None):
        if data is not None:
            if isinstance(data, list):
                data = data
            elif isinstance(data, types.GeneratorType):
                data = list(data)
            else:
                raise ValueError('data must be list or generator')
        elif data_path is not None:
            data = self.load_data(data_path)
        elif data_func is not None:
            data = self.data_func()
        else:
            raise ValueError(
                "The argument `data_path` or `data_func` is required.")
        self.data = list(self._to_observations(data))
        self.index = 0
        self.max_steps = len(self.data)
        self.max_price = max([obs.high for obs in self.data])
        self.history_num = history_num
        self.feature_num = Observation.N
        self.start_random = start_random
        self.ta_class = ta_class
        self.use_ta = use_ta

        if self.use_ta:
            self._load_data_with_ta(ta_timeperiods)

    def _load_data_with_ta(self, ta_timeperiods):
        if self.ta_class is None:
            from .ta import TaFeatures
            self.ta_class = TaFeatures
        self.ta_data = self.ta_class(self.data, timeperiods=ta_timeperiods)

    def _to_observations(self, data):
        for i, item in enumerate(data):
            item['index'] = i
            yield Observation(**item)

    def _load_json(self, path):
        with open(path) as f:
            data = json.load(f)
        return data

    def _load_pd(self, path):
        pass

    def load_data(self, path=None):
        filename, extension = os.path.splitext(path)
        if extension == '.json':
            data = self._load_json(path)
        return data

    @property
    def default_space(self):
        return [self.history_num + 1, self.feature_num]

    @property
    def first_price(self):
        return self.data[0].close

    @property
    def current_step(self):
        return self.data[self.index]

    @property
    def history(self):
        return self.data[:self.index]

    @property
    def recent_history(self):
        return History(self.data[self.index - self.history_num:self.index + 1],
                       history_num=self.history_num)

    @property
    def total(self):
        return self.history + [self.current_step]

    @property
    def ta_features(self):
        return self.ta_data.get_feature(self.index)

    def step(self):
        observation = self.current_step
        self.index += 1
        done = self.index >= self.max_steps
        return observation, done

    def reset(self, index=None):
        if not self.start_random:
            self.index = self.history_num + 1
        elif index is None:
            self.index = random.randint(self.history_num,
                                        int(self.max_steps * 0.9))
        else:
            self.index = index


class ExtraFeature:
    ex_obs_name = ['amount', 'buy_at', 'index', 'floating_rate', 'math_hour']

    def get_extra_features(self, info):
        features = {}
        features['amount'] = self.exchange.amount
        features['buy_at'] = info['buy_at'] / self.data.max_steps
        features['index'] = info['index'] / self.data.max_steps
        features['floating_rate'] = self.exchange.floating_rate
        features['math_hour'] = self.obs.math_hour

        return [features[name] for name in self.ex_obs_name]
