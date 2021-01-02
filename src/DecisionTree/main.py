import pandas as pd
import sklearn as skl
import sklearn.tree as tree
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer


def missing(data: pd.DataFrame):
    """
    Report the missing situation.

    :param data:
    :return:
    """

    missing_rate: pd.DataFrame = data.isnull().sum() / len(data)
    missing_rate = -(-missing_rate).sort_values()
    return missing_rate


def read_test_data():
    test_data_file_path_name = "../../data/speed_dating_test.csv"
    df = pd.read_csv(test_data_file_path_name)
    df = df.drop(columns=['uid'])
    y_pre = ((df['dec'] + df['dec_o']) / 2).values
    df = pd.concat([df, pd.DataFrame({'match': np.floor(y_pre)})], axis=1)
    print('--------------- test data init --------------------', df)
    return df


def calc_validation_accurcy(model, X_test, y_test):
    predict_test_lrc = model.predict(X_test)
    validate_accuracy = skl.metrics.accuracy_score(y_test, predict_test_lrc)
    print('Validation Accuracy:', validate_accuracy)


def read_data():
    """

    :return: Dataframe.
    """

    train_data_file_path_name = "../../data/speed_dating_train.csv"

    print("\033[0;32mRead Train Data:\033[0m \033[4m{filename}\033[0m".format(filename=train_data_file_path_name))

    df = pd.read_csv(train_data_file_path_name)

    print("Read finished.")
    print("\033[0;32mData Shape:\033[0m ", df.shape)
    print("\033[0;32mColumns:\033[0m ", df.columns)

    return df


def run_model(df: pd.DataFrame, model, print_result=True) -> (int, int):
    x = df.drop(columns=['match'])
    y = df['match']

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(model, x, y, cv=5)

    validate_accuracy = sum(scores) / len(scores)

    if print_result:
        print('Cross Validation Accuracy:', validate_accuracy)

    return 0, validate_accuracy


def do_vectorizer(data: pd.DataFrame, test_data: pd.DataFrame):
    char_params = [
        'goal', 'career_c',
    ]

    data[char_params] = data[char_params].astype('str')
    test_data[char_params] = test_data[char_params].astype('str')

    print(data[char_params].dtypes)

    vec = DictVectorizer(sparse=False)
    print('------ vectorizer ------')
    print(vec.fit_transform(data.to_dict(orient='records')))
    print('------ feature name ------', vec.get_feature_names())

    data['is_test'] = 0
    test_data['is_test'] = 1

    data_all = pd.concat([data, test_data], axis=0)

    print("--------- data all before vectorizer ----------", data_all)

    data_all = pd.DataFrame(vec.fit_transform(data_all.to_dict(orient='records')))
    data_all.columns = vec.get_feature_names()

    data = data_all.loc[data_all['is_test'] == 0]
    test_data = data_all.loc[data_all['is_test'] == 1]

    print("--------- test data after vectorizer ----------", test_data)
    print("--------- data after vectorizer ----------", data)

    return data, test_data


def feature_select(data: pd.DataFrame, test_data: pd.DataFrame):
    run_model(data, ExtraTreesClassifier())
    clf = ExtraTreesClassifier()

    from sklearn.feature_selection import SelectFromModel
    x = data.drop(columns=['match'])
    y = data['match']
    clf.fit(x, y)
    model = SelectFromModel(clf, prefit=True)
    x_new = model.transform(x)
    data = pd.concat([pd.DataFrame(x_new), data[['match']]], axis=1)

    print('--------- feature params -----------', model.get_params())

    test_data_x = test_data.drop(columns=['match'])
    test_data_y = np.array(test_data['match'])
    test_data_x_new = model.transform(test_data_x)
    test_data_y = pd.DataFrame({'match': test_data_y})
    print('--------- test data y after feature ---------', test_data_y)
    test_data = pd.concat([pd.DataFrame(test_data_x_new), test_data_y], axis=1)

    return data, test_data


def missing_handle(data: pd.DataFrame, test_data: pd.DataFrame):
    """

    :param data:
    :return:
    """
    data = data.drop(columns=['career', 'field', 'undergra', 'from', 'mn_sat', 'tuition', 'zipcode', 'income', 'dec', 'dec_o'])
    test_data = test_data.drop(columns=['career', 'field', 'undergra', 'from', 'mn_sat', 'tuition', 'zipcode', 'income', 'dec', 'dec_o'])

    from sklearn.impute import SimpleImputer
    from numpy import nan

    imputation_transformer1 = SimpleImputer(missing_values=nan, strategy="constant", fill_value=-1)

    data[data.columns] = imputation_transformer1.fit_transform(data[data.columns])
    test_data[test_data.columns] = imputation_transformer1.fit_transform(test_data[test_data.columns])

    print("\033[0;32mmissing rate before missing handle:\033[0m ", missing(data))
    print(data)
    data = data.dropna()
    test_data = test_data.dropna()
    print("\033[0;32mmissing rate after missing handle:\033[0m ", missing(data))
    print(data)

    return data, test_data


def main():
    test_data = read_test_data()
    data = read_data()

    print('---- test data shape ----', test_data.shape)

    data, test_data = missing_handle(data, test_data)
    data, test_data = do_vectorizer(data, test_data)
    data, test_data = feature_select(data, test_data)

    params = {
        'criterion': ['entropy'],
        'max_depth': [4, 7, 10, 13],
        'splitter': ['best', 'random'],
        'min_samples_split': [2, 4, 8, 16],
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'min_weight_fraction_leaf': [0, 0.1, 0.2, 0.3],
        # 'min_weight_fraction_leaf': [0],
        # 'max_features': ['log2'],
        'random_state': [0, 233, None],
        'min_impurity_decrease': [0, 0.1, 0.2, 0.3],
    }

    print('-------------test data-------------', test_data)

    class search_best_params:
        max_vc = 0
        best_params = {}
        case_total = 1
        case_finished = 0

        def __init__(self, params_dict: dict):
            for values in params_dict:
                self.case_total *= len(params_dict[values])

            self.case_finished = 0
            self.params_dict = params_dict

            print("----- case total: %d-----" % self.case_total)

        def exec(self):
            self.case_finished = 0
            self.best_params = {}
            self.run(params_dict=self.params_dict)

        def run(self, params_dict: dict):
            iter_ing = False

            for key in params_dict:
                v = params_dict[key]
                if type(v) == list:
                    for value in v:
                        new_dict = dict(params_dict)
                        new_dict[key] = value
                        self.run(new_dict)
                    iter_ing = True
                    break

            if not iter_ing:
                t_ac, v_ac = run_model(data, tree.DecisionTreeClassifier(**params_dict), print_result=False)
                self.case_finished += 1

                if v_ac > self.max_vc:
                    self.max_vc = v_ac
                    self.best_params = params_dict

                if self.case_finished * 1000 // self.case_total != \
                        (self.case_finished + 1) * 1000 // self.case_total:
                    print("done: %d/%d, " % (self.case_finished, self.case_total), self.max_vc, self.best_params)
    #
    # run_object = search_best_params(params)
    # run_object.exec()
    #
    # print(run_object.max_vc, run_object.best_params)

    best_params = {'criterion': 'entropy', 'max_depth': 10, 'splitter': 'random', 'min_samples_split': 4, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0, 'random_state': None, 'min_impurity_decrease': 0}
    best_params2 = {'criterion': 'gini', 'max_depth': 4, 'splitter': 'random', 'min_samples_split': 4, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0, 'max_features': None, 'random_state': None, 'min_impurity_decrease': 0}
    best_params3 = {'criterion': 'gini', 'max_depth': 7, 'splitter': 'random', 'min_samples_split': 16, 'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0, 'random_state': 0, 'min_impurity_decrease': 0}
    best_params4 = {'criterion': 'entropy', 'max_depth': 7, 'splitter': 'random', 'min_samples_split': 2, 'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0, 'random_state': 233, 'min_impurity_decrease': 0}
    run_model(data, tree.DecisionTreeClassifier(**best_params3))

    clf = tree.DecisionTreeClassifier(**best_params3)
    x = data.drop(columns=['match'])
    y = data['match']
    clf.fit(x, y)
    calc_validation_accurcy(clf, test_data.drop(columns=['match']), test_data['match'])


if __name__ == "__main__":
    main()


# best: {'criterion': 'entropy', 'max_depth': 7, 'splitter': 'best', 'min_samples_split': 8, 'min_samples_leaf': 16, 'min_weight_fraction_leaf': 0, 'max_features': 8}