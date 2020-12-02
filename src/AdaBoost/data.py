'''
数据清洗，
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *


def load_data(args, the_type):
    '''
    描述：读取数据
    参数：全局参数，类型(train, test, submit三种)
    返回：数据dataframe
    '''
    if the_type == 'train':
        place = args.data_place_train
    elif the_type == 'test':
        place = args.data_place_test
    elif the_type == 'submit':
        place = args.data_place_submit

    df = pd.read_csv(place, encoding = 'gbk')
    return df

def remove_useless_feature(args, data):
    '''
    描述：移除语义上和结果无关的feature
    参数：全局参数，训练数据
    返回：移除feature后的训练数据
    ''' 
    useless_feature = get_useless_columns()
    data = data.drop(useless_feature, axis = 1)
    return data

def remove_loss_feature(args, data):
    '''
    描述：输出并且移除丢失信息太多的feature
    参数：全局参数，训练数据
    返回：移除feature后的训练数据
    '''
    percent_missing = data.isnull().sum() / len(data)
    missing_value_df = pd.DataFrame({
    'percent_missing': percent_missing
    })
    missing_value_df.sort_values(by = 'percent_missing')
    print("Ratio of missing values:")
    print(missing_value_df)
    columns_removal = missing_value_df.index[missing_value_df.percent_missing > args.loss_removal_threshold].values
    print('-' * 100)
    print("The columns to be removed:")
    print(columns_removal)
    data = data.drop(columns_removal, axis = 1)
    print('-' * 100)
    print("Current data:")
    print(data)
    return data

def show_correleation(args, data):
    '''
    描述：求数据间的相关性，移除和match没啥关系的数据并且绘图
    参数：全局参数，数据
    返回：清理后的数据
    '''
    plt.subplots(figsize = (20, 30))
    axis= plt.axes()
    axis.set_title("Correlation Heatmap")
    corr = data.corr(method = 'spearman')

    columns_save = []
    for index in corr['match'].index.values:
        if abs(corr['match'][index]) >= args.corr_removal_threshold:
            columns_save.append(index)
    #columns_removal = data.index[abs(corr['match'][data.index]) < args.corr_removal_threshold].data
    print("The columns to be saved:")
    print(columns_save)
    data = data[columns_save]
    print("Current data:")
    print(data)
    corr = data.corr(method = 'spearman')
    sns.heatmap(corr,  xticklabels = corr.columns.values, cmap = "Blues")
    plt.show()
    return data

args = init_args()
data = load_data(args, 'train')
data = remove_useless_feature(args, data)
data = remove_loss_feature(args, data)
show_correleation(args, data)