from utils import *
import graphviz


def main():
    test_data = read_test_data()
    data = read_data()

    print('---- test data shape ----', test_data.shape)

    data, test_data = missing_handle(data, test_data)
    data, test_data = do_vectorizer(data, test_data)
    data, test_data = feature_select(data, test_data)

    print('-------------test data-------------', test_data)

    best_params3 = {'criterion': 'gini', 'max_depth': 3, 'splitter': 'random', 'min_samples_split': 16,
                    'min_samples_leaf': 4, 'min_weight_fraction_leaf': 0, 'random_state': 0, 'min_impurity_decrease': 0}
    best_params4 = {'criterion': 'entropy', 'max_depth': 3, 'splitter': 'random', 'min_samples_split': 2,
                    'min_samples_leaf': 8, 'min_weight_fraction_leaf': 0, 'random_state': 233,
                    'min_impurity_decrease': 0}

    run_model(data, tree.DecisionTreeClassifier(**best_params4))

    clf = tree.DecisionTreeClassifier(**best_params4)
    calc_validation_accurcy(clf, data.drop(columns=['match']), data['match'], test_data.drop(columns=['match']),
                            test_data['match'])

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=data.drop(columns=['match']).columns,
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    graph.view()

    params = {
        'criterion': ['entropy'],
        'max_depth': [4, 7, 10, 13],
        'splitter': ['best', 'random'],
        'min_samples_split': [2, 4, 8, 16],
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'min_weight_fraction_leaf': [0, 0.1, 0.2, 0.3],
        'random_state': [0, 233, None],
        'min_impurity_decrease': [0, 0.1, 0.2, 0.3],
    }

    from utils import search_best_params
    run_object = search_best_params(params)
    # run_object.exec()
    #
    # print(run_object.max_vc, run_object.best_params)


if __name__ == "__main__":
    main()

# best: {'criterion': 'entropy', 'max_depth': 7, 'splitter': 'best', 'min_samples_split': 8, 'min_samples_leaf': 16, 'min_weight_fraction_leaf': 0, 'max_features': 8}
