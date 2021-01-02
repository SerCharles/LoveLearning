from utils import *
from data import *
import pandas as pd
import time
import re
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pylab as plt


def output_result(args, y_result):
    """
    描述：输出结果到csv
    参数：全局参数，结果y
    返回：无
    """
    uid = []
    for i in range(len(y_result)):
        uid.append(i)

    result_df = pd.DataFrame({
        'uid': uid,
        'match': y_result
    })

    result_df.to_csv(args.grid_search_place, index=False)


def train_test(args, logger, x_train, y_train, x_test, y_test, n_estimators, max_depth, min_samples_split,
               min_samples_leaf, max_features):
    """
    训练测试模型
    :param args: 全局参数
    :param n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features: 超参数们
    :return: model, y_result, train_accuracy, test_accuracy, train_time, test_time
    """
    logger.debug("*" * 100)
    if n_estimators is None:
        n_estimators = 100
    if max_features is None:
        max_features = "auto"
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf, max_features=max_features)

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


def update_best_dictionary(model, result, train_accuracy, test_accuracy, train_time, test_time,
                           n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    """
    更新最优状态
    :param model, result, features, train_accuracy, test_accuracy, train_time, test_time: model params
    :param n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features: super params
    :return: 最优状态
    """
    best_dictionary = {'model': model, 'result': result, 'train_accuracy': train_accuracy,
                       'test_accuracy': test_accuracy, 'train_time': train_time, 'test_time': test_time,
                       'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf, "max_features": max_features}
    return best_dictionary


def print_best_dictionary(logger, best_dictionary):
    """
    描述：打印最佳情况
    参数：logger, 最佳情况
    返回：无
    """
    logger.debug("*" * 100)
    logger.debug("The Best Model:")
    logger.debug("Train Accuracy: {:.4}%".format(best_dictionary['train_accuracy']))
    logger.debug("Test Accuracy: {:.4}%".format(best_dictionary['test_accuracy']))
    logger.debug("Train Time: {:.4}s".format(best_dictionary['train_time']))
    logger.debug("Test Time: {:.4}s".format(best_dictionary['test_time']))

    logger.debug("Current Hyper Parameters: ")
    logger.debug("Random Forest Estimators: " + str(best_dictionary['n_estimators']))
    logger.debug("Trees Max Depth: " + str(best_dictionary['max_depth']))
    logger.debug("Trees Min Samples Split: " + str(best_dictionary['min_samples_split']))
    logger.debug("Trees Min Samples Leaf: " + str(best_dictionary['min_samples_leaf']))
    logger.debug("Trees Min Features: " + str(best_dictionary['max_features']))


def remove_feature(features, base_feature):
    """
    处理无用数据
    :param features: 参数列表
    :param base_feature: 需要去除的参数列表
    :return: 新的参数列表
    """
    new_features = []
    for string in features:
        if re.match(base_feature, string) is None:
            new_features.append(string)
    return new_features


def handle_data(args):
    """
    处理数据
    :param args:
    :param train_data:
    :param test_data:
    :return:
    """
    train_data = load_data(args, 'train')
    test_data = load_data(args, 'test')

    test_data = get_test_data_ground_truth(args, test_data)
    train_data = remove_useless_feature(args, train_data)
    test_data = remove_useless_feature(args, test_data)
    train_data = fill_loss_data(args, train_data)
    test_data = fill_loss_data(args, test_data)
    x_train = train_data.drop(['match'], axis=1)
    y_train = train_data[['match']]
    x_test = test_data.drop(['match'], axis=1)
    y_test = test_data[['match']]

    x_test = x_test.drop(['uid'], axis=1)

    return x_train, y_train, x_test, y_test


def random_forest_classify(args):
    """
    训练模型的主函数
    :param args: 全局参数
    :return: 无
    """
    logger = init_logging(args)
    best_acc = 0.0
    best_dictionary = {'model': None, 'result': None, 'train_accuracy': 0.0,
                       'test_accuracy': 0.0, 'train_time': 0.0, 'test_time': 0.0,
                       'n_estimators': -1, 'max_depth': -1, 'min_samples_split': -1,
                       'min_samples_leaf': -1, "max_features": -1}

    # 定义超参数组合
    n_estimatorses = [100]
    max_depths = [10]
    min_samples_splits = [2]
    min_samples_leaves = [1]
    max_features = ["sqrt"]

    # 遍历处理
    for n_estimators in n_estimatorses:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for min_samples_leaf in min_samples_leaves:
                    for max_feature in max_features:
                        x_train, y_train, x_test, y_test = handle_data(args)
                        model, result, train_accuracy, test_accuracy, train_time, test_time = train_test(
                            args, logger, x_train, y_train, x_test, y_test, n_estimators, max_depth,
                            min_samples_split, min_samples_leaf, max_feature)
                        if test_accuracy > best_acc:
                            best_acc = test_accuracy
                            best_dictionary = update_best_dictionary(model, result, train_accuracy, test_accuracy,
                                                                     train_time, test_time, n_estimators, max_depth,
                                                                     min_samples_split, min_samples_leaf,
                                                                     max_feature)
    print_best_dictionary(logger, best_dictionary)
    output_result(args, best_dictionary['result'])


if __name__ == '__main__':
    args = init_args()
    random_forest_classify(args)
