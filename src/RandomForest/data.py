import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *


def load_data(args, the_type):
    """
    读取数据
    :param args: 全局参数
    :param the_type: 'train' or 'test'
    :return: 数据dataframe
    """
    if the_type == 'train':
        place = args.data_place_train
    elif the_type == 'test':
        place = args.data_place_test
    df = pd.read_csv(place, encoding='gbk')
    return df


def remove_useless_feature(args, data):
    """
    移除语义上和结果无关的数据特征
    :param args: 全局参数
    :param data: 原数据
    :return: 处理后数据
    """
    useless_feature = get_useless_columns()
    data = data.drop(useless_feature, axis=1)
    return data


def remove_loss_feature(args, data):
    """
    移除丢失信息太多的数据特征
    :param args: 全局参数
    :param data: 原数据
    :return: 处理后数据
    """
    percent_missing = data.isnull().sum() / len(data)
    missing_value_df = pd.DataFrame({
        'percent_missing': percent_missing
    })
    missing_value_df.sort_values(by='percent_missing')
    columns_removal = missing_value_df.index[missing_value_df.percent_missing > args.loss_removal_threshold].values
    data = data.drop(columns_removal, axis=1)

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


def fill_loss_data(args, data):
    """
    补全数据丢失
    :param args:全局参数
    :param data: 训练数据
    :return: 补全后数据
    """
    if args.test:
        print("original data:")
        print(data)
    for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    if args.test:
        print("filled data:")
        print(data)
    return data


def get_test_data_ground_truth(args, test_data):
    """
    描述：计算出test数据的ground truth
    参数：全局参数，测试数据
    返回：测试数据
    """
    # 先求出test data的match

    test_data['match'] = (test_data['dec'].astype('int') & test_data['dec_o'].astype('int'))
    return test_data
