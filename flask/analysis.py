from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import MeanShift
import math

x_feature_names = ['GDP', 'Family', 'Health', 'Freedom', 'Corruption']


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
    print('\n(BY COLUMN) Y feature name : ', y_feature_name, '\n')
    classification_map = dict()
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
        report = classification_report(
            classifier=tree_classifier,
            x_train_rows=x_train_rows,
            y_train_rows=y_train_rows,
            y_feature_name=y_feature_name
        )
        classification_map[value_column] = report

    for key, val in classification_map.items():
        print('\tColumn value : ', key)
        print('\t\tMost importance : ')
        for k, v in val['most_importance'].items():
            print('\t\t\t', k, ' = ', v)
        print('\t\tImportance : ')
        for k, v in val['x_features'].items():
            print('\t\t\t', k, ' = ', v)
        print('\t\tY feature statistics : ')
        for k, v in val['y_feature'].items():
            print('\t\t\t', k, ' = ', v)
    return classification_map


def classification_by_world(concat_data_frame, y_feature_name):
    print('\n(BY WORLD) Y feature name : ', y_feature_name, '\n')
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
    print('\t\tMost importance : ')
    for k, v in report['most_importance'].items():
        print('\t\t\t', k, ' = ', v)
    print('\t\tImportance : ')
    for k, v in report['x_features'].items():
        print('\t\t\t', k, ' = ', v)
    print('\t\tY feature statistics : ')
    for k, v in report['y_feature'].items():
        print('\t\t\t', k, ' = ', v)
    return report
