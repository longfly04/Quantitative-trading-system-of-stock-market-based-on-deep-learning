"""
使用基准的LSTM模型构建全流程预测模型
"""

from __future__ import absolute_import

from .backend import *
from .backend import backend as K
import os
import datetime as dt
import numpy as np 
import pickle
from numpy import newaxis
import datetime as dt 
from keras import activations, constraints, initializers, regularizers
from keras.layers import *
from keras import Model
from keras.models import Sequential, load_model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard

from utils.tools import *


class LSTM_Model(Model):
    def __init__(self, config, name=None, **kwargs):
        super(LSTM_Model, self).__init__(**kwargs)
        self.model_cfg = config['model']['lstm']
        self.train_cfg = config['training']
        self.pre_cfg = config['preprocess']
        self.data_cfg = config['data']
        self.predict_cfg = config['prediction']
        self.name = name

    @info
    def build_model(self, input_shape, output_shape, epoch_steps):
        self.inputs_shape = input_shape
        self.outputs_shape = output_shape
        self.model = Sequential()
        self.epoch_steps = epoch_steps
        assert len(epoch_steps) == 2
        # epoch_steps : (steps_per_epoch, validation_steps)
        
        for layer in self.model_cfg['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            inputs = self.inputs_shape if 'input_layer' in layer else (None, None)
            outputs = self.outputs_shape[0] if 'output_layer' in layer else None

            if layer['type'] == 'dense':
                if outputs is not None:
                    self.model.add(Dense(outputs, activation=activation))
                else:
                    self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                if inputs is not None:
                    self.model.add(LSTM(neurons, input_shape=inputs, return_sequences=return_seq))
                else:
                    self.model.add(LSTM(neurons, return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'flatten':
                self.model.add(Flatten())

        self.model.compile(loss=self.model_cfg['loss'], optimizer=self.model_cfg['optimizer'], metrics=['accuracy'])
        self.model.summary()
        
        if self.model_cfg['plot_model']:
            from keras.utils import plot_model
            plot_model(self.model, to_file=self.model_cfg['plot_model_path'])

        print('[Model] Model Compiled')

    @info
    def load_model_weight(self, model_file=None):
        '''
        从已经训练好的模型中载入模型
        '''
        print('[Model] Loading model from file %s' % model_file)
        self.model = load_model(model_file)

    @info
    def train_model(self, x, y, save_model=True, end_date=None):
        '''
        使用普通方法训练，x,y都是batched data
        '''
        print('[Model] Training Started')

        callbacks = [
            TensorBoard(log_dir=self.train_cfg['tensorboard_dir']),
            ]

        self.history = self.model.fit(
                                      x,
                                      y,
                                      epochs=self.train_cfg['epochs'],
                                      batch_size=self.train_cfg['batch_size'],
                                      callbacks=callbacks,
                                      )

        epoch_loss = self.history.history['loss'][-1]
        epoch_val_loss = self.history.history['val_loss'][-1]
        epoch_acc = self.history.history['acc'][-1]
        epoch_val_acc = self.history.history['val_acc'][-1]
        if save_model:
            if not os.path.exists(self.train_cfg['save_model_path']): 
                os.makedirs(self.train_cfg['save_model_path'])
            loss_str = str(epoch_loss)[:6] + str(epoch_val_loss)[:6]
            acc_str = str(epoch_acc)[:6] + str(epoch_val_acc)[:6]
            stock_name = self.name
            save_fname = os.path.join(self.train_cfg['save_model_path'],
                                 '%s-%s-%s-%s-%s.h5' % (dt.datetime.now().strftime('%Y%m%d_%H%M%S'),
                                               loss_str, acc_str, stock_name, end_date))
            self.model.save(save_fname)
            print('[Saving] Model saved as %s' % save_fname)
        print('[Model] Training Completed.')

        return epoch_loss, epoch_val_loss, epoch_acc, epoch_val_acc

    @info
    def train_model_generator(self, xy_gen, val_gen, save_model=True, end_date=None):
        '''
        使用迭代器输入数据。
        '''
        print('[Model] Generator Training Started')

        callbacks = [
            # EarlyStopping(monitor='val_loss', patience=2),
            # ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
            TensorBoard(log_dir=self.train_cfg['tensorboard_dir']),
            ]

        self.history = self.model.fit_generator(
                                                xy_gen,
                                                steps_per_epoch=self.epoch_steps[0],
                                                epochs=self.train_cfg['epochs'],
                                                callbacks=callbacks,
                                                validation_data=val_gen,
                                                validation_steps=self.epoch_steps[1],
                                                validation_freq=self.train_cfg['validation_freq'],
                                                )

        epoch_loss = self.history.history['loss'][-1]
        epoch_val_loss = self.history.history['val_loss'][-1]
        epoch_acc = self.history.history['acc'][-1]
        epoch_val_acc = self.history.history['val_acc'][-1]
        if save_model:
            if not os.path.exists(self.train_cfg['save_model_path']): 
                os.makedirs(self.train_cfg['save_model_path'])
            loss_str = str(epoch_loss)[:6] + str(epoch_val_loss)[:6]
            acc_str = str(epoch_acc)[:6] + str(epoch_val_acc)[:6]
            stock_name = self.name
            save_fname = os.path.join(self.train_cfg['save_model_path'],
                                 '%s-%s-%s-%s-%s.h5' % (dt.datetime.now().strftime('%Y%m%d_%H%M%S'),
                                               loss_str, acc_str, stock_name, end_date))
            self.model.save(save_fname)
            print('[Saving] Model saved as %s' % save_fname)
        print('[Model] Generator Training Completed.')

        return epoch_loss, epoch_val_loss, epoch_acc, epoch_val_acc


    @info
    def predict_seqence(self, pred_x, save_result=True):
        """
        使用普通方式预测序列，这种预测方式可能是因为PCA的原因引入了未来信息造成的信息泄露，
        精度比想象中高，趋势把握的很准，很奇怪。
        """
        batch_size = self.train_cfg['batch_size']
        predict_type = self.pre_cfg['predict_type']
        res_list = []
        for x in pred_x:
            ret = self.model.predict(x[newaxis, :, :,], batch_size=batch_size, verbose=1,)
            res_list.append(ret)

        if save_result:
            if not os.path.exists(self.predict_cfg['save_result_path']): 
                os.makedirs(self.predict_cfg['save_result_path'])
            now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = 'results_'+ now + self.name +'.pkl'
            save_fname = os.path.join(self.predict_cfg['save_result_path'],
                                      save_file)
            with open(save_fname, 'wb') as out:
                pickle.dump(res_list, out)
            print("[Saving] Results is saved as \'%s\' ." %save_fname)

        return res_list


    @info
    def predict_future_generated(self, predict_gen, steps=1, save_result=True):
        """
        使用生成器预测未来数据
        """
        batch_size = self.train_cfg['batch_size']
        predict_type = self.pre_cfg['predict_type']

        ret_list = self.model.predict_generator(predict_gen, steps=steps, verbose=1,)
        print("[Predict] Predict result is as follows...")
        print(ret_list)

        if save_result:
            if not os.path.exists(self.predict_cfg['save_result_path']): 
                os.makedirs(self.predict_cfg['save_result_path'])
            now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = 'results_'+ now + self.name +'.pkl'
            save_fname = os.path.join(self.predict_cfg['save_result_path'],
                                      save_file)
            with open(save_fname, 'wb') as out:
                pickle.dump(ret_list, out)
            print("[Saving] Results is saved as \'%s\' ." %save_fname)

        return ret_list

    @info
    def predict_one_step(self, pred_x, save_result=False):
        """
        预测未来的未知数据，使用现有数据集的最后一个窗口预测未来数据

        参数：
            date_step:对应的日期步
            pred_x:特征
            save：保存选项
        """
        batch_size = self.train_cfg['batch_size']
        predict_type = self.pre_cfg['predict_type']
        
        ret = self.model.predict(pred_x[newaxis, :, :,], batch_size=batch_size, verbose=1,)

        if save_result:
            if not os.path.exists(self.predict_cfg['save_result_path']): 
                os.makedirs(self.predict_cfg['save_result_path'])
            now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = 'results_'+ now + self.name +'.pkl'
            save_fname = os.path.join(self.predict_cfg['save_result_path'],
                                      save_file)
            with open(save_fname, 'wb') as out:
                pickle.dump(ret, out)
            print("[Saving] Results is saved as \'%s\' ." %save_fname)

        return ret