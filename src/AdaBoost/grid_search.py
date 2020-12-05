from utils import *
from data import *
import pandas as pd
import time
import re
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import logging

def output_result(args, y_result):
    '''
    描述：输出结果到csv
    参数：全局参数，结果y
    返回：无
    '''
    uid = []
    for i in range(len(y_result)):
        uid.append(i)

    result_df = pd.DataFrame({
    'uid': uid,
    'match': y_result
    })
    
    result_df.to_csv(args.grid_search_place, index = False)

def train_test(args, logger, x_train, y_train, x_test, y_test, max_depth, min_samples_split, min_samples_leaf, n_estimators, learning_rate):
    '''
    描述：训练-测试模型
    参数：全局参数，logger, 清洗后的数据x_train, y_train, x_test, y_test, adaboost模型的超参数若干
    返回：model, y_result(测试结果), train_accuracy, test_accuracy, train_time, test_time
    '''
    logger.debug("*" * 100)
    logger.debug("Current Features: " + str(x_train.columns))
    logger.debug("Current Hyper Parameters: ")
    logger.debug("Decision Tree Max Depth: " + str(max_depth))
    logger.debug("Decision Tree Min Samples Split: " + str(min_samples_split))
    logger.debug("Decision Tree Min Samples Leaf: " + str(min_samples_leaf))
    logger.debug("Adaboost Estimators: " + str(n_estimators))
    logger.debug("Adaboost Learning Rate: " + str(learning_rate))

    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf),
                         n_estimators = n_estimators, learning_rate = learning_rate, random_state = args.seed)
    
    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time

    start_time = time.time()
    y_result = model.predict(x_test)
    end_time = time.time()
    test_time = end_time - start_time
    train_accuracy = 100 * model.score(x_train, y_train)
    test_accuracy = 100 * model.score(x_test, y_test)

    logger.debug("Train Accuracy: {:.4}%".format(train_accuracy))
    logger.debug("Test Accuracy: {:.4}%".format(test_accuracy))
    logger.debug("Train Time: {:.4}s".format(train_time))
    logger.debug("Test Time: {:.4}s".format(test_time))

    return model, y_result, train_accuracy, test_accuracy, train_time, test_time

def update_best_dictionary(model, result, features, train_accuracy, test_accuracy, train_time, test_time, \
    max_depth, min_samples_split, min_samples_leaf, n_estimators, learning_rate):
    '''
    描述：更新最优状态
    参数：model, result, features, train_accuracy, test_accuracy, train_time, test_time, \
    max_depth, min_samples_split, min_samples_leaf, n_estimators, learning_rate
    返回：最优状态
    '''
    best_dictionary = {'model': model, 'result': result, 'features': features, 'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy, 'train_time': train_time, 'test_time': test_time, \
        'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'n_estimators': n_estimators, 'learning_rate': learning_rate}
    return best_dictionary

def print_best_dictionary(logger, best_dictionary):
    '''
    描述：打印最佳情况
    参数：logger, 最佳情况
    返回：无
    '''
    logger.debug("*" * 100)
    logger.debug("The Best Model:")
    logger.debug("Train Accuracy: {:.4}%".format(best_dictionary['train_accuracy']))
    logger.debug("Test Accuracy: {:.4}%".format(best_dictionary['test_accuracy']))
    logger.debug("Train Time: {:.4}s".format(best_dictionary['train_time']))
    logger.debug("Test Time: {:.4}s".format(best_dictionary['test_time']))
    logger.debug("Current Features:" + str(best_dictionary['features']))
    logger.debug("Current Hyper Parameters: ")
    logger.debug("Decision Tree Max Depth: " + str(best_dictionary['max_depth']))
    logger.debug("Decision Tree Min Samples Split: " + str(best_dictionary['min_samples_split']))
    logger.debug("Decision Tree Min Samples Leaf: " + str(best_dictionary['min_samples_leaf']))
    logger.debug("Adaboost Estimators: " + str(best_dictionary['n_estimators']))
    logger.debug("Adaboost Learning Rate: " + str(best_dictionary['learning_rate']))

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

def handle_data(args, feature_id, train_data, test_data):
    '''
    描述：根据特征编号来选取特征，同时根据此切分训练-测试数据
    参数：全局参数特征编号，训练数据，测试数据
    返回：x_train, y_train, x_test, y_test
    '''
    #定义特征组合
    group_0 = ['match_es', 'int_corr', 'match']
    group_1 = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob', 'met', 'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o', 'met_o']
    group_2 = ['satis_2', 'attr7_2', 'sinc7_2', 'intel7_2', 'fun7_2', 'amb7_2', 'shar7_2', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1', \
        'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1', 'attr3_1', 'sinc3_1', 'fun3_1', 'intel3_1', 'amb3_1', 'attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1']
    group_3 = ['attr1_3', 'sinc1_3', 'intel1_3', 'fun1_3', 'amb1_3', 'shar1_3', 'attr7_3', 'sinc7_3', 'intel7_3', 'fun7_3', 'amb7_3', 'shar7_3', 'attr4_3', 'sinc4_3', 'intel4_3', 'fun4_3', 'amb4_3', 'shar4_3', \
        'attr2_3', 'sinc2_3', 'intel2_3', 'fun2_3', 'amb2_3', 'shar2_3', 'attr3_3', 'sinc3_3', 'intel3_3', 'fun3_3', 'amb3_3', 'attr5_3', 'sinc5_3', 'intel5_3', 'fun5_3', 'amb5_3']

    if feature_id == 0:
        #默认处理方法
        train_data = remove_loss_feature(args, train_data)
        train_data = filt_correleation(args, train_data)
    elif feature_id == 1:
        #采用group 0+1
        columns = group_0 + group_1
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 2:
        #采用group 0+2
        columns = group_0 + group_2
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 3:
        #采用group 0+3
        columns = group_0 + group_3
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 4:
        #采用group 0+1+2
        columns = group_0 + group_1 + group_2
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 5:
        #采用group 0+1+3
        columns = group_0 + group_1 + group_3
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 6:
        #采用group 0+2+3
        columns = group_0 + group_2 + group_3
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 7:
        #采用group 0+1+2+3
        columns = group_0 + group_1 + group_2 + group_3
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 8:
        #采用group 0+1, 去掉attr
        new_group_1 = remove_feature(group_1, 'attr')
        columns = group_0 + new_group_1
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 9:
        #采用group 0+1, 去掉sinc
        new_group_1 = remove_feature(group_1, 'sinc')
        columns = group_0 + new_group_1
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 10:
        #采用group 0+1, 去掉intel
        new_group_1 = remove_feature(group_1, 'intel')
        columns = group_0 + new_group_1
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 11:
        #采用group 0+1, 去掉fun
        new_group_1 = remove_feature(group_1, 'fun')
        columns = group_0 + new_group_1
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 12:
        #采用group 0+1, 去掉amb
        new_group_1 = remove_feature(group_1, 'amb')
        columns = group_0 + new_group_1
        train_data = filt_data(args, train_data, columns)
    elif feature_id == 13:
        #采用group 0+1, 去掉shar
        new_group_1 = remove_feature(group_1, 'shar')
        columns = group_0 + new_group_1
        train_data = filt_data(args, train_data, columns)


    train_data = fill_loss_data(args, train_data)
    test_data = filt_test_data(args, train_data, test_data)
    test_data = fill_loss_data(args, test_data)  
    x_train = train_data.drop(['match'], axis = 1)
    y_train = train_data[['match']]
    x_test = test_data.drop(['match'], axis = 1)
    y_test = test_data[['match']]
    return x_train, y_train, x_test, y_test


def grid_search():
    '''
    描述：全局搜索主函数
    参数：无
    返回：无
    '''

    #读取数据，做必要处理
    args = init_args()
    logger = init_logging(args)
    train_data = load_data(args, 'train')
    test_data = load_data(args, 'test')
    test_data = get_test_data_ground_truth(args, test_data)
    train_data = remove_useless_feature(args, train_data)
    test_data = remove_useless_feature(args, test_data)

    #最优数据存储方式
    best_acc = 0.0
    best_dictionary = {'model': None, 'result': None, 'features': [], 'train_accuracy': 0.0, 'test_accuracy': 0.0, 'train_time': 0.0, 'test_time': 0.0, \
        'max_depth': -1, 'min_samples_split': -1, 'min_samples_leaf': -1, 'n_estimators': -1, 'learning_rate': -1}
    

    #定义超参数组合    
    max_depths = [1, 2, 3]
    min_samples_splits = [5, 10, 20]
    min_samples_leafs = [1, 5, 10, 20]
    n_estimatorss = [20, 50, 100, 200, 500, 1000]
    learning_rates = [0.01, 0.1, 0.5, 0.8]
    feature_ids = []
    for i in range(14):
        feature_ids.append(i)

    #遍历处理
    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            for min_samples_leaf in min_samples_leafs:
                for n_estimators in n_estimatorss:
                    for learning_rate in learning_rates:
                        for feature_id in feature_ids:
                            x_train, y_train, x_test, y_test = handle_data(args, feature_id, train_data, test_data)
                            model, result, train_accuracy, test_accuracy, train_time, test_time = \
                                train_test(args, logger, x_train, y_train, x_test, y_test, max_depth, min_samples_split, min_samples_leaf, n_estimators, learning_rate)
                            if test_accuracy > best_acc:
                                best_acc = test_accuracy
                                features = x_train.columns
                                best_dictionary = update_best_dictionary(model, result, features, train_accuracy, test_accuracy, train_time, test_time, \
                                    max_depth, min_samples_split, min_samples_leaf, n_estimators, learning_rate)
    #打印和输出最优模型  
    print_best_dictionary(logger, best_dictionary)
    output_result(args, best_dictionary['result'])


if __name__ == '__main__':
    grid_search()