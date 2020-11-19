import os

import predict
import selector
import analysis
import pandas

from flask import Flask
from flask import request
from flask import jsonify

# '2015' => './2015.csv'
map_year_and_file_path = dict()

# '2015' => <data_frame>
map_year_and_data_frame = dict()

# 'East Europe' => <row_data_frame>
map_region_and_rows = dict()

# 'Russia' => <row_data_frame>
map_country_and_rows = dict()

# 'Russia' => 'East Europe'
map_country_and_region = dict()

# Объединение
concat_data_frame = pandas.DataFrame

# --Анализ--------------------------------------------------------------------------------------------------------------

# Задача классификации для параметра Ранг
cl_region_rank, cl_world_rank, mi_region_rank = dict(), dict(), dict()

# Задача классификации для параметра Оценка респондента
cl_region_score, cl_world_score, mi_region_score = dict(), dict(), dict()

# Задача классификации для стран с параметром Ранг
cl_country_rank, mi_country_rank = dict(), dict()

# Задача классификации для стран с параметром Оценка респондента
cl_country_score, mi_country_score = dict(), dict()


# ----------------------------------------------------------------------------------------------------------------------


def create_data_frames():
    global concat_data_frame, \
        map_country_and_region, \
        map_year_and_file_path, \
        map_year_and_data_frame, \
        map_region_and_rows, \
        map_country_and_rows
    for key, path in map_year_and_file_path.items():
        map_year_and_data_frame[key] = pandas.read_csv(path)
        map_year_and_data_frame[key]['Rank'] = [
            abs(x - map_year_and_data_frame[key]['Rank'].count()) for x in map_year_and_data_frame[key]['Rank']
        ]
    if len(map_year_and_data_frame) == 0 or len(map_year_and_data_frame) != len(map_year_and_file_path):
        raise Exception('Дата фреймы не сформированы.')
    map_country_and_region = selector.add_region_if_not_exist(map_year_and_data_frame)
    concat_data_frame = selector.concat(map_year_and_data_frame)
    map_region_and_rows = selector.group_by_region(concat_data_frame)
    map_country_and_rows = selector.group_by_country(concat_data_frame)
    return


def find_data_files():
    global map_year_and_file_path, map_year_and_data_frame
    root_py = os.path.dirname(os.path.abspath(__file__)).split('\\flask')[0]
    for dir_obj in os.walk(root_py):
        dir_path = str(dir_obj[0])
        if 'data' and 'input' in dir_path:
            print(dir_path)
            for file in dir_obj[2]:
                file_name = str(file)
                if '.csv' in file_name:
                    map_year_and_file_path[file_name.split('.')[0]] = os.path.join(dir_path, file)
            break
    if len(map_year_and_file_path) == 0:
        raise Exception('Файлов содержащих данные не найденно.')
    return


def init():
    global cl_region_rank, \
        cl_world_rank, \
        mi_region_rank, \
        cl_region_score, \
        cl_world_score, \
        mi_region_score, \
        cl_country_rank, \
        mi_country_rank, \
        cl_country_score, \
        mi_country_score
    #
    find_data_files()
    create_data_frames()
    #
    cl_region_rank = analysis.classification_by_column(map_region_and_rows, 'Rank')
    cl_region_score = analysis.classification_by_column(map_region_and_rows, 'Score')
    #
    cl_world_rank = analysis.classification_by_world(concat_data_frame, 'Rank')
    cl_world_score = analysis.classification_by_world(concat_data_frame, 'Score')
    #
    cl_country_rank = analysis.classification_by_column(map_country_and_rows, 'Rank')
    cl_country_score = analysis.classification_by_column(map_country_and_rows, 'Score')
    return


init()

# -routing---------------------------------------------------------------------------------------------------------------

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict/classification', methods=['POST'])
def predict_classification():
    scope = request.args.get('scope')
    y_feature_name = request.args.get('y')
    x_predict_features = request.json
    if scope == 'world':
        if y_feature_name == 'rank':
            result = predict.classification_by_world(
                x_predict_values=x_predict_features,
                cl_by_world=cl_world_rank,
                y_feature_name='Rank',
                concat_data_frame=concat_data_frame
            )
            return jsonify(result), 200
        if y_feature_name == 'score':
            result = predict.classification_by_world(
                x_predict_values=x_predict_features,
                cl_by_world=cl_world_score,
                y_feature_name='Score',
                concat_data_frame=concat_data_frame
            )
            return jsonify(result), 200
        return jsonify({'error': 'bad params'}), 400
    if scope == 'region':
        region = request.args.get('region')
        if y_feature_name == 'rank':
            result = predict.classification_by_column(
                value_column=region,
                cl_by_column=cl_region_rank,
                x_predict_values=x_predict_features,
                map_column_and_rows=map_region_and_rows,
                y_feature_name='Rank',
                flag='region'
            )
            return jsonify(result), 200
        if y_feature_name == 'score':
            result = predict.classification_by_column(
                value_column=region,
                cl_by_column=cl_region_score,
                x_predict_values=x_predict_features,
                map_column_and_rows=map_region_and_rows,
                y_feature_name='Score',
                flag='region'
            )
            return jsonify(result), 200
        return jsonify({'error': 'bad params'}), 400
    if scope == 'country':
        country = request.args.get('country')
        if y_feature_name == 'rank':
            result = predict.classification_by_column(
                value_column=country,
                cl_by_column=cl_country_rank,
                x_predict_values=x_predict_features,
                map_column_and_rows=map_country_and_rows,
                y_feature_name='Rank',
                flag='country'
            )
            return jsonify(result), 200
        if y_feature_name == 'score':
            result = predict.classification_by_column(
                value_column=country,
                cl_by_column=cl_country_score,
                x_predict_values=x_predict_features,
                map_column_and_rows=map_country_and_rows,
                y_feature_name='Score',
                flag='country'
            )
            return jsonify(result), 200
        return jsonify({'error': 'bad params'}), 400
    return jsonify({'error': 'bad params'}), 400


if __name__ == '__main__':
    app.run()
