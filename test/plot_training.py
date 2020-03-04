import matplotlib.pyplot as plt 
import seaborn as sb

import pandas as pd 
import numpy as np 

import os,sys
sys.path.insert(0, 'D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning')

from utils.tools import search_file

def plot_training(data):
    """"""
    plt.figure(figsize=(16, 10), dpi=150)
    x = pd.DatetimeIndex(data['predict_date'])
    data = data.set_index(x)
    plt.subplot(2, 1, 1)
    plt.title('Training and Validation ')
    plt.plot(data['epoch_acc'], label='Training Accuracy')
    plt.plot(data['epoch_val_acc'], label='Validation Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(data['epoch_loss'], label='Training Loss',)
    plt.plot(data['epoch_val_loss'], label='Validation Loss',)
    plt.legend()

    plt.show()

def load_data(file_path):
    """"""
    data = pd.read_csv(file_path)

    plot_data = data[['predict_date', 'epoch_loss','epoch_val_loss','epoch_acc','epoch_val_acc']]

    return plot_data

def main():
    for path in search_file('D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning\\saved_results', '.csv'):
        data = load_data(path)
        plot_training(data)


if __name__ == '__main__':
    main()