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
    参数：全局参数，类型(train, test两种)
    返回：数据dataframe
    '''
    if the_type == 'train':
        place = args.data_place_train
    elif the_type == 'test':
        place = args.data_place_test

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
    columns_removal = missing_value_df.index[missing_value_df.percent_missing > args.loss_removal_threshold].values
    data = data.drop(columns_removal, axis = 1)

    if args.test == 1:
        print("Ratio of missing values:")
        print(missing_value_df)
        print('-' * 100)
        print("The columns to be removed:")
        print(columns_removal)
        print('-' * 100)
        print("Current data:")
        print(data)
    return data

def filt_correleation(args, data):
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
    data = data[columns_save]
    corr = data.corr(method = 'spearman')

    if args.test:
        print("The columns to be saved:")
        print(columns_save)
        print("Current data:")
        print(data)
        sns.heatmap(corr,  xticklabels = corr.columns.values, cmap = "Blues")
        plt.show()
    return data

def filt_test_data(args, train_data, test_data):
    '''
    描述：根据训练数据处理方式，处理测试数据
    参数：全局参数，训练数据，测试数据
    返回：测试数据
    '''
    #先求出test data的match
    test_data['match'] = (test_data['dec'] & test_data['dec_o'])
    columns = train_data.columns
    test_data = test_data[columns]
    if args.test:
        print(test_data)
    return test_data


def fill_loss_data(args, data):
    '''
    描述：补全训练数据/测试数据的缺失
    参数：全局参数，训练数据
    返回：补全后的数据
    '''
    if args.test:
        print("original data:")
        print(data)
    for column in data.columns:
        data[column].fillna(data[column].mode()[0],inplace = True)
    if args.test:
        print("filled data:")
        print(data)
    return data
