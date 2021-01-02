import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


# 载入数据
def load_data():
    data_train_frame = pd.read_csv("../../data/speed_dating_train.csv")
    data_test_frame = pd.read_csv("../../data/speed_dating_test.csv")
    return data_train_frame, data_test_frame


# 语义无关列的清除
def get_irrelevant_data():
    return ['dec_o', 'dec', 'iid', 'id', 'gender', 'idg',
            'condtn', 'wave', 'round', 'position', 'positin1',
            'order', 'partner', 'pid', 'field', 'tuition', 'career']


# 数据集中清除无关特征
def remove_irrelevant_data(data):
    irrelevant_data = get_irrelevant_data()
    data = data.drop(irrelevant_data, axis=1)
    return data


# ground truth
def get_ground_truth(data):
    """
    描述：get ground truth for the data
    :param data: 全局参数，测试数据
    :return: 测试数据
    """
    data['match'] = (data['dec'].astype("bool") & data['dec_o'].astype("bool"))
    return data


# 处理缺省值-数据清洗
def handle_missing(data, percent):
    percent_missing = data.isnull().sum() / len(data)
    missing_df = pd.DataFrame({'column_name': data.columns, 'percent_missing': percent_missing})
    missing_show = missing_df.sort_values(by='percent_missing')
    print(missing_show[missing_show['percent_missing'] > 0].count())
    print('----------------------------------')
    print(missing_show[missing_show['percent_missing'] > percent])
    columns = missing_show.index[missing_show['percent_missing'] > percent]
    data = data.drop(columns=columns, axis=1)
    return data


# 特征列选取
def select_data(data, columns):
    data = data[columns]
    return data


# 补全样本缺失值
def fill_loss_data(data):
    data = data.copy(deep=False)
    for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    return data


# 分析特征相关性
def display_corr(data):
    plt.subplots(figsize=(20, 15))
    axis = plt.axes()
    axis.set_title("Correlation HeatMap")
    corr = data.corr(method="spearman")
    columns_save = []
    for index in corr['match'].index.values:
        if abs(corr['match'][index]) >= 0.1:
            columns_save.append(index)
    data = data[columns_save]
    corr = data.corr(method='spearman')
    sns.heatmap(corr, xticklabels=corr.columns.values, annot=True)
    plt.show()


def remove_feature(features, base_feature):
    '''
    描述：从一群特征中去除某些特征（比如取消所有sinc/attr这种）
    参数：特征列表，你要去除的
    返回：新的特征列表
    '''
    new_features = []
    for string in features:
        if re.match(base_feature, string) == None:
            new_features.append(string)
    return new_features


# 特征分组
def corr_feature(feature_id):
    # 保留相关系数0.15以上
    # 定义特征组合
    group_0 = ['match']
    group_1 = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob', 'met', 'attr_o', 'sinc_o', 'intel_o',
               'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o', 'met_o']
    group_2 = ['satis_2', 'attr7_2', 'sinc7_2', 'intel7_2', 'fun7_2', 'amb7_2', 'shar7_2', 'attr1_1', 'sinc1_1',
               'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1',
               'shar4_1', \
               'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1', 'attr3_1', 'sinc3_1', 'fun3_1',
               'intel3_1', 'amb3_1', 'attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1']
    group_3 = ['attr1_3', 'sinc1_3', 'intel1_3', 'fun1_3', 'amb1_3', 'shar1_3', 'attr7_3', 'sinc7_3', 'intel7_3',
               'fun7_3', 'amb7_3', 'shar7_3', 'attr4_3', 'sinc4_3', 'intel4_3', 'fun4_3', 'amb4_3', 'shar4_3', \
               'attr2_3', 'sinc2_3', 'intel2_3', 'fun2_3', 'amb2_3', 'shar2_3', 'attr3_3', 'sinc3_3', 'intel3_3',
               'fun3_3', 'amb3_3', 'attr5_3', 'sinc5_3', 'intel5_3', 'fun5_3', 'amb5_3']
    if feature_id == 1:
        # 采用group 0+1
        columns = group_0 + group_1
    elif feature_id == 2:
        # 采用group 0+2
        columns = group_0 + group_2
    elif feature_id == 3:
        # 采用group 0+3
        columns = group_0 + group_3
    elif feature_id == 4:
        # 采用group 0+1+2
        columns = group_0 + group_1 + group_2
    elif feature_id == 5:
        # 采用group 0+1+3
        columns = group_0 + group_1 + group_3
    elif feature_id == 6:
        # 采用group 0+2+3
        columns = group_0 + group_2 + group_3
    elif feature_id == 7:
        # 采用group 0+1+2+3
        columns = group_0 + group_1 + group_2 + group_3
    elif feature_id == 8:
        # 采用group 0+1, 去掉attr
        new_group_1 = remove_feature(group_1, 'attr')
        columns = group_0 + new_group_1
    elif feature_id == 9:
        # 采用group 0+1, 去掉sinc
        new_group_1 = remove_feature(group_1, 'sinc')
        columns = group_0 + new_group_1
    elif feature_id == 10:
        # 采用group 0+1, 去掉intel
        new_group_1 = remove_feature(group_1, 'intel')
        columns = group_0 + new_group_1
    elif feature_id == 11:
        # 采用group 0+1, 去掉fun
        new_group_1 = remove_feature(group_1, 'fun')
        columns = group_0 + new_group_1
    elif feature_id == 12:
        # 采用group 0+1, 去掉amb
        new_group_1 = remove_feature(group_1, 'amb')
        columns = group_0 + new_group_1
    elif feature_id == 13:
        # 采用group 0+1, 去掉shar
        new_group_1 = remove_feature(group_1, 'shar')
        columns = group_0 + new_group_1
    return columns


if __name__ == '__main__':
    train_data, test_data = load_data()
    print(train_data.columns)
    train_data = handle_missing(train_data, 0.7)
    train_data = remove_irrelevant_data(train_data)
    train_data = fill_loss_data(train_data)
    # display_corr(data=train_data)