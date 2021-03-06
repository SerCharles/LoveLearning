'''
常用的函数集合
'''

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def init_args():
    '''
    描述：初始化全局参数
    参数：无
    返回：全局参数args
    '''
    parser = argparse.ArgumentParser(description = "Arguments of the project.")
    #数据读取相关
    parser.add_argument("--seed", type = int, default = 1111)
    parser.add_argument("--data_dir", type = str, default = os.path.join('..', '..', 'data'), help = "The directory of datas")
    parser.add_argument("--data_name_train", type = str, default = 'speed_dating_train.csv', help = "The file name of train data")
    parser.add_argument("--data_name_test", type = str, default = 'speed_dating_test.csv', help = "The file name of test data")
    parser.add_argument("--data_name_submit", type = str, default = 'sample_submission.csv', help = "The file name of submit data")
    parser.add_argument("--result_dir", type = str, default = os.path.join('..', '..', 'result'), help = "The directory of results")
    parser.add_argument("--ground_truth_name", type = str, default = 'ground_truth.csv', help = "The file name of ground truth")
    parser.add_argument("--result_name", type = str, default = 'result.csv', help = "The file name of result")
    parser.add_argument("--grid_search_name", type = str, default = 'grid_search.csv', help = "The file name of result")
    parser.add_argument("--log_name", type = str, default = 'log.log', help = "The file name of result")

    #是否输出测试结果和显示可视化结果，0否1是
    parser.add_argument('--test', type = int, default = 0)

    #各种丢弃特征的threshold
    parser.add_argument("--loss_removal_threshold", type = float, default = 0.5)
    parser.add_argument("--corr_removal_threshold", type = float, default = 0.1)
    
    #算法参数
    parser.add_argument("--max_depth", type = int, default = 1)
    parser.add_argument("--min_samples_split", type = int, default = 5)
    parser.add_argument("--min_samples_leaf", type = int, default = 20)
    parser.add_argument("--n_estimators", type = int, default = 200)
    parser.add_argument("--learning_rate", type = float, default = 0.5)
    
    #是否复现最优结果
    parser.add_argument("--best", type = int, default = 0)

    args = parser.parse_args()

    #计算得到文件位置
    args.data_place_train = os.path.join(args.data_dir, args.data_name_train)
    args.data_place_test = os.path.join(args.data_dir, args.data_name_test)
    args.data_place_submit = os.path.join(args.data_dir, args.data_name_submit)
    args.ground_truth_place = os.path.join(args.result_dir, args.ground_truth_name)
    args.result_place = os.path.join(args.result_dir, args.result_name)
    args.grid_search_place = os.path.join(args.result_dir, args.grid_search_name)
    args.log_place = os.path.join(args.result_dir, args.log_name)
    return args

def init_logging(args):
    '''
    描述：创建logging
    参数：全局参数
    返回：logger
    '''
    # 创建Logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 创建Handler

    # 终端Handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    # 文件Handler
    fileHandler = logging.FileHandler(args.log_place, mode = 'w', encoding = 'UTF-8')
    fileHandler.setLevel(logging.NOTSET)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # 添加到Logger中
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger


def get_useless_columns():
    '''
    描述：返回没有意义的特征
    参数：无
    返回：一个数组，没有意义的特征
    '''
    return ['dec_o', 'dec', 'iid', 'id', 'gender', 'idg', 'condtn', 'wave', 'round', 'position', 'positin1', 'order', 'partner', 'pid', 'field', 'tuition', 'career']