from utils import *
from data import *
import pandas as pd
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def handle_data(args, train_data, test_data):
    '''
    描述：清洗数据
    参数：全局参数，训练数据, 测试数据
    返回：清洗后的数据x_train, y_train, x_test, y_test
    '''
    test_data = get_test_data_ground_truth(args, test_data)
    train_data = remove_useless_feature(args, train_data)
    test_data = remove_useless_feature(args, test_data)
    train_data = remove_loss_feature(args, train_data)
    train_data = filt_correleation(args, train_data)
    train_data = fill_loss_data(args, train_data)
    test_data = filt_test_data(args, train_data, test_data)
    test_data = fill_loss_data(args, test_data)    
    x_train = train_data.drop(['match'], axis = 1)
    y_train = train_data[['match']]
    x_test = test_data.drop(['match'], axis = 1)
    y_test = test_data[['match']]
    return x_train, y_train, x_test, y_test

def train_test(args, x_train, y_train, x_test, y_test):
    '''
    描述：训练-测试模型
    参数：全局参数，清洗后的数据x_train, y_train, x_test, y_test
    返回：模型，结果
    '''
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth = args.max_depth, min_samples_split = args.min_samples_split, min_samples_leaf = args.min_samples_leaf),
                    n_estimators = args.n_estimators, learning_rate = args.learning_rate, random_state = args.seed)

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
    print("Train Accuracy: {:.4}%".format(train_accuracy))
    print("Test Accuracy: {:.4}%".format(test_accuracy))
    print("Train Time: {:.4}s".format(train_time))
    print("Test Time: {:.4}s".format(test_time))

    return model, y_result

def output_result(args, y_test, y_result):
    '''
    描述：输出结果到csv
    参数：全局参数，测试y(ground truth)，结果y
    返回：无
    '''
    uid = []
    for i in range(len(y_test['match'])):
        uid.append(i)
    ground_truth_df = pd.DataFrame({
    'uid': uid,
    'match': y_test['match']
    })
    result_df = pd.DataFrame({
    'uid': uid,
    'match': y_result
    })
    
    ground_truth_df.to_csv(args.ground_truth_place, index = False)
    result_df.to_csv(args.result_place, index = False)



if __name__ == '__main__':    
    args = init_args()
    train_data = load_data(args, 'train')
    test_data = load_data(args, 'test')
    x_train, y_train, x_test, y_test = handle_data(args, train_data, test_data)
    model, y_result = train_test(args, x_train, y_train, x_test, y_test)
    output_result(args, y_test, y_result)