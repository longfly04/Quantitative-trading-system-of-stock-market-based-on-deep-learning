# 一个基准方案：使用LSTM预测。

from __future__ import absolute_import

from .backend import *
from .backend import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.layers import *
from keras import Model
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import os
import datetime as dt
from utils.tools import *
from numpy import newaxis
import datetime as dt 

class LSTM_Model(Model):
    def __init__(self, config, **kwargs):
        super(LSTM_Model, self).__init__(**kwargs)
        self.model_cfg = config['model']['baseline']
        self.training_cfg = config['training']
        self.date_cfg = config['data']
        self.after_cfg = config['after_training']

    @info
    def build_model(self, ):

        self.model = Sequential()
        
        for layer in self.model_cfg['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'flatten':
                self.model.add(Flatten())

        self.model.compile(loss=self.model_cfg['loss'], optimizer=self.model_cfg['optimizer'])
        self.model.summary()
        
        if self.model_cfg['plot_model']:
            from keras.utils import plot_model
            plot_model(self.model, to_file=self.model_cfg['plot_model_path'])

        print('[Model] Model Compiled')

    @info
    def load_model(self, model_file=None):
        '''
        从已经训练好的模型中载入模型
        '''
        print('[Model] Loading model from file %s' % model_file)
        self.model = load_model(model_file)

    @info
    def train_model(self, x, y, val_data):
        '''
        使用普通方法训练
        '''
        print('[Model] Training Started')

        save_fname = os.path.join(self.training_cfg['save_path'],
                                 '%s-e%s.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), 
                                 str(self.training_cfg['epochs'])))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
            TensorBoard(log_dir=self.training_cfg['tensorboard_dir'])
        ]

        x = x[x.shape[0]%self.training_cfg['batch_size']:]
        y = y[y.shape[0]%self.training_cfg['batch_size']:]

        self.history = self.model.fit(
                                      x,
                                      y,
                                      epochs=self.training_cfg['epochs'],
                                      batch_size=self.training_cfg['batch_size'],
                                      callbacks=callbacks,
                                      validation_data=val_data,
                                      )
        if self.training_cfg['save']:
            if not os.path.exists(self.training_cfg['save_dir']): 
                os.makedirs(self.training_cfg['save_dir'])
            self.model.save(save_fname)
            print('[Saving] Model saved as %s' % save_fname)
        print('[Model] Training Completed.')

    @info
    def train_model_generator(self, xy_gen, val_gen, save_model=True):
        '''
        使用迭代器输入数据。
        '''
        print('[Model] Generator Training Started')

        callbacks = [
            # EarlyStopping(monitor='val_loss', patience=2),
            # ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
            TensorBoard(log_dir=self.training_cfg['tensorboard_dir'])
        ]


        self.history = self.model.fit_generator(
                                                xy_gen,
                                                steps_per_epoch=self.training_cfg['steps_per_epoch'],
                                                epochs=self.training_cfg['epoch_per_mask'],
                                                # callbacks=callbacks,
                                                validation_data=val_gen,
                                                validation_steps=self.training_cfg['steps_per_val'],
                                                validation_freq=self.training_cfg['validation_freq']
                                                )
        if save_model:
            if not os.path.exists(self.training_cfg['save_model_path']): 
                os.makedirs(self.training_cfg['save_model_path'])
            epoch_loss = self.history.history['loss'][-1]
            epoch_val_loss = self.history.history['val_loss'][-1]
            loss_str = 'loss_' + str(epoch_loss)[:6] + '-val_loss_' + str(epoch_val_loss)[:6]

            save_fname = os.path.join(self.training_cfg['save_model_path'],
                                 '%s-%s.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'),
                                               loss_str))
            self.model.save(save_fname)
            print('[Saving] Model saved as %s' % save_fname)
        print('[Model] Generator Training Completed.')

    @info
    def predict_submit(self, predict_embedding, save_result=True):
        import numpy as np 
        import pickle

        flow_max = self.date_cfg['flow_max']
        pred_result = dict()
        for k in predict_embedding.keys():
            pred_x = predict_embedding[k]
            k_result = []
            for i in pred_x:
                ret = self.model.predict(i[newaxis, :, :,], batch_size=32, verbose=1,)
                ret = flow_max * ret
                k_result.append(ret)

            k_result = np.array(k_result).reshape((-1, pred_x.shape[1], 1))
            pred_result[k] = k_result

        if save_result:
            if not os.path.exists(self.after_cfg['save_result_path']): 
                os.makedirs(self.after_cfg['save_result_path'])
            now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = 'results_'+ now +'.pkl'
            save_fname = os.path.join(self.after_cfg['save_result_path'],
                                      save_file)
            with open(save_fname, 'wb') as out:
                pickle.dump(pred_result, out)
            print("[Saving] Results is saved as \' %s \' ." %save_fname)

        return pred_result

