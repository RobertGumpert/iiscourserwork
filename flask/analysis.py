from itertools import chain
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, f_regression
import math

x_feature_names = ['GDP', 'Family', 'Health', 'Freedom', 'Corruption']


def euclidean_distance(a, b):
    summary = 0
    for i in range(len(a)):
        difference = a[i] - b[i]
        power = pow(difference, 2)
        summary += power
    return summary ** (1 / float(2))


def base_statistics(rows):
    base = dict()
    dispersion_columns = rows.std()
    for column_name, dispersion in dispersion_columns.iteritems():
        average = int(rows[column_name].mean())
        if math.isnan(average):
            average = 0
        min = int(rows[column_name].min())
        if math.isnan(min):
            average = 0
        max = int(rows[column_name].max())
        if math.isnan(max):
            average = 0
        if math.isnan(dispersion):
            average = 0
        base[column_name] = dict(
            average=float(average),
            min=float(min),
            max=float(max),
            dispersion=float(dispersion)
        )
    return base


def select_min_max_average(data_series):
    min_value = data_series.min()
    max_value = data_series.max() + 1
    average_value = int((sum(data_series.tolist()) / len(data_series.tolist())))
    return min_value, max_value, average_value


def select_two_classes(y_feature_rows):
    y_min, y_max, y_average = select_min_max_average(y_feature_rows)
    y_train = []
    for i, value in y_feature_rows.items():
        if value <= y_average:
            y_train.append(y_min)
        else:
            y_train.append(y_max)
    return y_train


def select_third_classes(y_feature_rows):
    y_min, y_max, y_aver = select_min_max_average(y_feature_rows)
    y_train = []
    for i in range(y_feature_rows.size):
        if i <= int(y_feature_rows.size / 3):
            y_train.append(y_min)
            continue
        if int(y_feature_rows.size / 3) < i <= int((2 * y_feature_rows.size) / 3):
            y_train.append(y_aver)
            continue
        y_train.append(y_max)
    return y_train


def classification(x_train_rows, y_train_rows):
    tree_classifier = DecisionTreeClassifier(max_depth=4)
    tree_classifier.fit(x_train_rows, y_train_rows)
    return tree_classifier


def classification_report(classifier, x_train_rows, y_train_rows, y_feature_name):
    report = dict()
    importance_value = classifier.feature_importances_
    x_train_base = base_statistics(x_train_rows)
    y_train_base = base_statistics(y_train_rows)
    importance_of_features = dict()
    max_importance_value = importance_value.max()
    max_importance_feature = ""
    for feature_name, value_importance in zip(x_train_rows.columns, importance_value):
        if value_importance == max_importance_value:
            max_importance_feature = feature_name
        importance_of_features[feature_name] = dict(
            feature_importance=float(value_importance),
            feature_max=x_train_base[feature_name]['max'],
            feature_min=x_train_base[feature_name]['min'],
            feature_dispersion=x_train_base[feature_name]['dispersion'],
            feature_average=x_train_base[feature_name]['average']
        )
    report['most_importance'] = dict(
        feature_name=max_importance_feature,
        feature_importance=float(max_importance_value)
    )
    report['x_features'] = importance_of_features
    report['y_feature'] = dict(
        feature_importance=y_feature_name,
        feature_max=y_train_base[y_feature_name]['max'],
        feature_min=y_train_base[y_feature_name]['min'],
        feature_dispersion=y_train_base[y_feature_name]['dispersion'],
        feature_average=y_train_base[y_feature_name]['average']
    )
    report['classifier'] = classifier
    return report


def classification_by_column(map_column_and_rows, y_feature_name):
    # print('\n(BY COLUMN) Y feature name : ', y_feature_name, '\n')
    report = dict()
    for value_column, rows in map_column_and_rows.items():
        if rows is None:
            continue
        rows_copy = rows.copy()
        rows_copy = rows_copy.sort_values(by=[y_feature_name])
        x_train_rows = rows_copy.loc[:, x_feature_names]
        rows_copy[y_feature_name] = rows_copy[y_feature_name].astype('int')
        y_train_rows = rows_copy.loc[:, [y_feature_name]]
        if x_train_rows.size == 0 or y_train_rows.size == 0:
            continue
        y_train_list = select_two_classes(rows_copy[y_feature_name])
        tree_classifier = classification(x_train_rows, y_train_list)
        cl_report = classification_report(
            classifier=tree_classifier,
            x_train_rows=x_train_rows,
            y_train_rows=y_train_rows,
            y_feature_name=y_feature_name
        )
        report[value_column] = cl_report

    # for key, val in report.items():
    #     print('\tColumn value : ', key)
    #     print('\t\tMost importance : ')
    #     for k, v in val['most_importance'].items():
    #         print('\t\t\t', k, ' = ', v)
    #     print('\t\tImportance : ')
    #     for k, v in val['x_features'].items():
    #         print('\t\t\t', k, ' = ', v)
    #     print('\t\tY feature statistics : ')
    #     for k, v in val['y_feature'].items():
    #         print('\t\t\t', k, ' = ', v)
    return report


def classification_by_world(concat_data_frame, y_feature_name):
    # print('\n(BY WORLD) Y feature name : ', y_feature_name, '\n')
    concat_copy = concat_data_frame.copy()
    concat_copy.sort_values(by=[y_feature_name])
    x_train_rows = concat_copy.loc[:, x_feature_names]
    concat_copy[y_feature_name] = concat_copy[y_feature_name].astype('int')
    y_train_rows = concat_copy.loc[:, [y_feature_name]]
    y_train_list = select_third_classes(concat_copy[y_feature_name])
    tree_classifier = classification(x_train_rows, y_train_list)
    report = classification_report(
        classifier=tree_classifier,
        x_train_rows=x_train_rows,
        y_train_rows=y_train_rows,
        y_feature_name=y_feature_name
    )
    # print('\t\tMost importance : ')
    # for k, v in report['most_importance'].items():
    #     print('\t\t\t', k, ' = ', v)
    # print('\t\tImportance : ')
    # for k, v in report['x_features'].items():
    #     print('\t\t\t', k, ' = ', v)
    # print('\t\tY feature statistics : ')
    # for k, v in report['y_feature'].items():
    #     print('\t\t\t', k, ' = ', v)
    return report


def clustering_kmeans(concat_data_frame):
    concat_copy = concat_data_frame.copy()
    #
    x_train_rows = concat_copy.loc[:, x_feature_names]
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(x_train_rows)
    concat_copy = concat_copy.loc[concat_copy['Year'] == 2019]
    #
    countries_in_cluster = dict()
    for i, _ in enumerate(kmeans.cluster_centers_):
        countries_in_cluster[i] = list()
    #
    countries = dict()
    for row in concat_copy.iterrows():
        distances = list()
        features = row[1].loc[x_feature_names]
        features = features.tolist()
        for i, _ in enumerate(kmeans.cluster_centers_):
            distance = euclidean_distance(kmeans.cluster_centers_[i], features)
            distances.insert(i, distance)
        countries[row[1]['Country']] = distances
    #
    for country, distances in countries.items():
        min_index = min(range(len(distances)), key=distances.__getitem__)
        countries_in_cluster[min_index].append(dict(
            country=country,
            distance=distances[min_index]
        ))
    #
    report = dict(
        clucterizator=kmeans,
        centers=kmeans.cluster_centers_,
        countries=countries_in_cluster
    )
    #
    return report


def happiness_models(concat_data_frame):
    y_feature_name = 'Score'
    test_range = 0.3
    train_range = 0.7
    alpha_default = 1.0
    #
    concat_copy = concat_data_frame.copy()
    x_all_rows = concat_copy.loc[:, x_feature_names]
    y_all_rows = concat_copy.loc[:, [y_feature_name]]
    #
    x_list = x_all_rows.values.tolist()
    y_list = y_all_rows.values.tolist()
    train_x, test_x, train_y, test_y = train_test_split(
        x_list, y_list,
        test_size=test_range,
        train_size=train_range
    )

    #
    def regularization_linear_regression(train_x, test_x):
        def percents_list(frame_y):
            max_y = max(frame_y)
            per_list = [((y[0] / max_y[0]) * 100) for y in frame_y]
            return per_list

        #
        train_percents_list_y = percents_list(train_y)
        test_percents_list_y = percents_list(test_y)
        #
        model = linear_model.LinearRegression()
        model.fit(train_x, train_percents_list_y)
        model_score = model.score(test_x, test_percents_list_y)
        #
        lasso = linear_model.Lasso(alpha=alpha_default)
        lasso.fit(train_x, train_percents_list_y)
        lasso_score = lasso.score(test_x, test_percents_list_y)
        #
        ridge = linear_model.Ridge(alpha=alpha_default)
        ridge.fit(train_x, train_percents_list_y)
        ridge_score = ridge.score(test_x, test_percents_list_y)
        #
        if lasso_score > ridge_score and lasso_score > model_score:
            return lasso, train_percents_list_y, test_percents_list_y
        if ridge_score > lasso_score and ridge_score > model_score:
            return ridge, train_percents_list_y, test_percents_list_y
        return model, train_percents_list_y, test_percents_list_y

    #
    def regularization_perceptron(train_x, test_x):

        def boolean_list(frame_y):
            list_y = [y[0] for y in frame_y]
            aver = float(sum(list_y) / float(len(list_y)))
            boolean_list_y = list()
            for y in list_y:
                if y <= aver:
                    boolean_list_y.append(0)
                if y > aver:
                    boolean_list_y.append(1)
            return boolean_list_y

        #
        train_boolean_list_y = boolean_list(train_y)
        test_boolean_list_y = boolean_list(test_y)
        #
        model = linear_model.Perceptron()
        model.fit(train_x, train_boolean_list_y)
        model_score = model.score(test_x, test_boolean_list_y)
        #
        lasso = linear_model.Perceptron(penalty='l1', alpha=alpha_default)
        lasso.fit(train_x, train_boolean_list_y)
        lasso_score = lasso.score(test_x, test_boolean_list_y)
        #
        ridge = linear_model.Perceptron(penalty='l2', alpha=alpha_default)
        ridge.fit(train_x, train_boolean_list_y)
        ridge_score = ridge.score(test_x, test_boolean_list_y)
        #
        if lasso_score > ridge_score and lasso_score > model_score:
            return lasso, train_boolean_list_y, test_boolean_list_y
        if ridge_score > lasso_score and ridge_score > model_score:
            return ridge, train_boolean_list_y, test_boolean_list_y
        return model, train_boolean_list_y, test_boolean_list_y

    train_percents_list_y, test_percents_list_y, train_boolean_list_y, test_boolean_list_y = list(), list(), list(), list()

    #
    def ranging(lr_model, perceptron):
        ranks = dict()

        def rank_to_dict(rs, names):
            rs = np.abs(rs)
            rs = map(lambda x: round(x, 2), rs)
            return dict(zip(names, rs))

        def meaning(rs, names):
            mean_dict = dict()
            models = len(rs.keys())
            for name in names:
                summary = 0.0
                for model, rank in rs.items():
                    summary += rank[name]
                mean_dict[name] = summary / float(models)
            mean_dict = dict(sorted(mean_dict.items(), key=lambda item: item[1]))
            return mean_dict

        #
        lr_rfe = RFE(lr_model)
        lr_rfe.fit(train_x, train_percents_list_y)
        ranks['lr'] = rank_to_dict(lr_rfe.ranking_, x_feature_names)
        #
        per_rfe = RFE(perceptron)
        per_rfe.fit(train_x, train_boolean_list_y)
        ranks['pr'] = rank_to_dict(per_rfe.ranking_, x_feature_names)
        #
        mean = meaning(ranks, x_feature_names)
        c = 0
        importance_features = list()
        for key, _ in mean.items():
            if c > (int(len(mean) / 2) - 1):
                importance_features.append(key)
            c += 1
        return importance_features

    first_train_lr_model, train_percents_list_y, test_percents_list_y = regularization_linear_regression(train_x,
                                                                                                         test_x)
    first_train_perceptron, train_boolean_list_y, test_boolean_list_y = regularization_perceptron(train_x, test_x)
    most_importance_features = ranging(first_train_lr_model, first_train_perceptron)
    #
    x_all_rows = concat_copy.loc[:, most_importance_features]
    y_all_rows = concat_copy.loc[:, [y_feature_name]]
    #
    x_list = x_all_rows.values.tolist()
    y_list = y_all_rows.values.tolist()
    train_x, test_x, train_y, test_y = train_test_split(
        x_list, y_list,
        test_size=test_range,
        train_size=train_range
    )
    #
    final_lr_model, _, _ = regularization_linear_regression(train_x, test_x)
    final_perceptron, _, _ = regularization_perceptron(train_x, test_x)
    report = dict(
        lr=final_lr_model,
        pr=final_perceptron,
        importance=most_importance_features
    )
    return report
