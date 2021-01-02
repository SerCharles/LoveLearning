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


def calc_validation_accurcy(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
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
