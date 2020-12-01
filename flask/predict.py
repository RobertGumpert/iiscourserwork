from random import randrange

import pandas
import statistics

x_feature_names = ['GDP', 'Family', 'Health', 'Freedom', 'Corruption']


def classification_by_column(cl_by_column, map_column_and_rows, value_column, x_predict_values, y_feature_name, flag):
    column_report = cl_by_column[value_column]
    if cl_by_column is None:
        return None
    data = []
    result = dict()
    for column, value in x_predict_values.items():
        if column not in x_feature_names:
            return None
        data.append(value)
    x_predict = pandas.DataFrame([data], columns=x_feature_names)
    cl = column_report['classifier']
    predict_class = int(cl.predict(x_predict))
    result['most_importance'] = column_report['most_importance']
    result['statistics'] = column_report['x_features']
    result['y'] = column_report['y_feature']
    result['classes'] = cl.classes_.tolist()
    result['predict_class'] = predict_class
    if flag == 'country':
        if predict_class == cl.classes_[0]:
            result['class'] = 'Станет менее счастливой.'
        else:
            result['class'] = 'Станет более счастливой.'
    if flag == 'region':
        if predict_class == cl.classes_[0]:
            result['class'] = 'Наименее счастливые страны в регионе.'
        else:
            result['class'] = 'Наиболее счастливые страны в регионе.'
        predict_samples = map_column_and_rows[value_column].loc[
            (map_column_and_rows[value_column]['Year'] == 2019)
        ]
        predict_samples = predict_samples.drop_duplicates(subset='Country')
        result['predict'] = predict_samples[['Country', 'Region', y_feature_name]].to_dict('records')
    return result


def classification_by_world(cl_by_world, concat_data_frame, x_predict_values, y_feature_name):
    data = []
    result = dict()
    for column, value in x_predict_values.items():
        if column not in x_feature_names:
            return None
        data.append(value)
    cl = cl_by_world['classifier']
    x_predict = pandas.DataFrame([data], columns=x_feature_names)
    predict_class = int(cl.predict(x_predict))
    result['predict_class'] = predict_class
    if predict_class == cl.classes_[0]:
        result['class'] = 'Наименее счастливые страны в мире.'
    if predict_class == cl.classes_[1]:
        result['class'] = 'Средне счастливые страны в мире.'
    if predict_class == cl.classes_[2]:
        result['class'] = 'Наиболее счастливые страны в мире.'
    predict_samples = concat_data_frame.loc[
        (concat_data_frame['Year'] == 2019)
    ]
    predict_samples = predict_samples.drop_duplicates(subset='Country')
    result['classes'] = cl.classes_.tolist()
    result['most_importance'] = cl_by_world['most_importance']
    result['statistics'] = cl_by_world['x_features']
    result['y'] = cl_by_world['y_feature']
    result['predict'] = predict_samples[['Country', 'Region', y_feature_name]].to_dict('records')
    return result


def clustering_kmeans(kmeans_clustering, x_predict_values):
    data = []
    result = dict()
    for column, value in x_predict_values.items():
        if column not in x_feature_names:
            return None
        data.append(value)
    kmeans = kmeans_clustering['clucterizator']
    x_predict = pandas.DataFrame([data], columns=x_feature_names)
    cluster = kmeans.predict(x_predict)[0]
    cluster_countries = kmeans_clustering['countries'][cluster]
    country_index = randrange(len(cluster_countries))
    result['country'] = cluster_countries[country_index]['country']
    similar = list()
    for _ in range(0, 5):
        similar_country = randrange(len(cluster_countries))
        similar.append(cluster_countries[similar_country]['country'])
    result['similar'] = similar
    return result


def happiness_models(hp_models, x_predict_values):
    result = dict()
    data = []
    for column, value in x_predict_values.items():
        if column in hp_models["importance"]:
            data.append(value)
    lr_score = hp_models["lr"].predict([data])
    pr_score = hp_models["pr"].predict([data])
    result['lr_score'] = lr_score
    result['pr_score'] = pr_score
    result['importance'] = hp_models["importance"]
    return result
