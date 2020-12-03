'''
常用的函数集合
'''

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    #是否输出测试结果和显示可视化结果，0否1是
    parser.add_argument('--test', type = int, default = 0)

    #各种丢弃特征的threshold
    parser.add_argument("--loss_removal_threshold", type = float, default = 0.5)
    parser.add_argument("--corr_removal_threshold", type = float, default = 0.1)
    
    #算法参数
    parser.add_argument("--max_depth", type = int, default = 2)
    parser.add_argument("--min_samples_split", type = int, default = 20)
    parser.add_argument("--min_samples_leaf", type = int, default = 5)
    parser.add_argument("--n_estimators", type = int, default = 200)
    parser.add_argument("--learning_rate", type = float, default = 0.8)

    args = parser.parse_args()

    #计算得到文件位置
    args.data_place_train = os.path.join(args.data_dir, args.data_name_train)
    args.data_place_test = os.path.join(args.data_dir, args.data_name_test)
    args.data_place_submit = os.path.join(args.data_dir, args.data_name_submit)
    args.ground_truth_place = os.path.join(args.result_dir, args.ground_truth_name)
    args.result_place = os.path.join(args.result_dir, args.result_name)
    return args

def get_useless_columns():
    '''
    描述：返回没有意义的特征
    参数：无
    返回：一个数组，没有意义的特征
    '''
    return ['dec_o', 'dec', 'iid', 'id', 'gender', 'idg', 'condtn', 'wave', 'round', 'position', 'positin1', 'order', 'partner', 'pid', 'field', 'tuition', 'career']