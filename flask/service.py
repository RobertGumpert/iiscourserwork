import datetime
import os
import selector
import analysis
import pandas
import predict
import configparser
from fpdf import FPDF
import matplotlib.pyplot as plt

# --Tg Bot Configs------------------------------------------------------------------------------------------------------

tg_configs = None

root_py = None

# --Агрегированные данные-----------------------------------------------------------------------------------------------

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

#
kmeans_clustering = dict()

#
hp_models = dict()


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
        max_rank = map_year_and_data_frame[key]['Rank'].max()
        map_year_and_data_frame[key]['Rank'] = [
            abs(x - max_rank) for x in map_year_and_data_frame[key]['Rank']
        ]
    if len(map_year_and_data_frame) == 0 or len(map_year_and_data_frame) != len(map_year_and_file_path):
        raise Exception('Дата фреймы не сформированы.')
    map_country_and_region = selector.add_region_if_not_exist(map_year_and_data_frame)
    concat_data_frame = selector.concat(map_year_and_data_frame)
    map_region_and_rows = selector.group_by_region(concat_data_frame)
    map_country_and_rows = selector.group_by_country(concat_data_frame)
    return


def find_data_files():
    global map_year_and_file_path, map_year_and_data_frame, tg_configs, root_py
    root_py = os.path.dirname(os.path.abspath(__file__))
    tg_configs = configparser.ConfigParser()
    tg_configs.read('configs.ini')
    project_dir = root_py.split('\\flask')[0]
    for dir_obj in os.walk(project_dir):
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
        mi_country_score, \
        kmeans_clustering, \
        hp_models
    #
    find_data_files()
    create_data_frames()

    # find_similar_country('Russia')

    cl_region_rank = analysis.classification_by_column(map_region_and_rows, 'Rank')
    cl_region_score = analysis.classification_by_column(map_region_and_rows, 'Score')
    #
    cl_world_rank = analysis.classification_by_world(concat_data_frame, 'Rank')
    cl_world_score = analysis.classification_by_world(concat_data_frame, 'Score')
    #
    cl_country_rank = analysis.classification_by_column(map_country_and_rows, 'Rank')
    cl_country_score = analysis.classification_by_column(map_country_and_rows, 'Score')
    #
    kmeans_clustering = analysis.clustering_kmeans(concat_data_frame)
    hp_models = analysis.happiness_models(concat_data_frame)

    create_pdf()

    return


def find_similar_country(explored_country):
    if explored_country not in map_country_and_rows:
        raise Exception("This country isn't exist. ")
    #
    file_name_rank = os.path.dirname(__file__) + os.path.normpath(f'/reports/plots/{explored_country}_rank.png')
    file_name_score = os.path.dirname(__file__) + os.path.normpath(f'/reports/plots/{explored_country}_score.png')
    #
    if (os.path.isfile(file_name_rank) is False) or (os.path.isfile(file_name_score) is False):
        print('New graphs')

        def get_last_year_data(data_frame):
            last_explored_year = data_frame.loc[
                                 :, ['Year']
                                 ].values.tolist()[-1][0]
            select_data_frame = data_frame.loc[
                data_frame['Year'] == last_explored_year
                ]
            explored_params = select_data_frame.loc[
                              :, analysis.x_feature_names
                              ].values.tolist()[0]
            return last_explored_year, select_data_frame, explored_params

        #
        explored_country_year, explored_country_data_frame, explored_country_params = get_last_year_data(
            data_frame=map_country_and_rows[explored_country]
        )
        #
        distance_result = dict()
        life_level = dict()
        life_level[explored_country] = dict(
            rank=explored_country_data_frame['Rank'].values.tolist()[0],
            score=explored_country_data_frame['Score'].values.tolist()[0]
        )
        #
        for country, data_frame_country in map_country_and_rows.items():
            if country == explored_country:
                continue
            #
            last_year, last_year_data_frame, last_year_params = get_last_year_data(
                data_frame=map_country_and_rows[country]
            )
            #
            if len(last_year_params) == 0:
                continue
            #
            distance = analysis.euclidean_distance(
                a=explored_country_params,
                b=last_year_params
            )
            #
            distance_result[country] = distance
            life_level[country] = dict(
                rank=last_year_data_frame['Rank'].values.tolist()[0],
                score=last_year_data_frame['Score'].values.tolist()[0]
            )
        #
        distance_result = dict(sorted(distance_result.items(), key=lambda item: item[1]))

        #
        def create_plot(deep, feature, file_name, xl, yl):
            i = 0
            x = list()
            y = list()
            for country, value_distance in distance_result.items():
                if i == 0:
                    x.append(0)
                    y.append(life_level[explored_country][feature])
                    plt.annotate(explored_country, (0, life_level[explored_country][feature]))
                if i == deep:
                    i = 0
                    break
                x.append(value_distance)
                y.append(life_level[country][feature])
                plt.annotate(country, (value_distance, life_level[country][feature]))
                i += 1
            plt.scatter(x, y)
            plt.xlabel(xl)
            plt.ylabel(yl)
            plt.savefig(file_name)
            plt.clf()

        deep_search = 15
        #
        create_plot(
            deep=deep_search,
            feature='score',
            file_name=file_name_score,
            xl=f'Насколько страна близка к {explored_country} по ценностям',
            yl='Уровень счастья по мнению жителей'
        )
        create_plot(
            deep=deep_search,
            feature='rank',
            file_name=file_name_rank,
            xl=f'Насколько страна близка к {explored_country} по ценностям',
            yl='Уровень счастья по рейтингу'
        )
    #
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Similar countries for {explored_country}", ln=1, align="C")
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 10, txt=f"First graph of similar countries for {explored_country} and param rank", ln=1)
    pdf.image(file_name_rank, x=50, y=50, w=100)
    pdf.cell(0, 10, txt=f"Second graph of similar countries for {explored_country} and param score", ln=1)
    pdf.image(file_name_score, x=50, y=160, w=100)
    bytes_list = pdf.output(name='report.pdf', dest='S')
    pdf.output(name='report.pdf')
    #
    return bytes_list


def create_pdf():
    pdf = FPDF()
    pdf.set_font("Arial", size=9)
    col_width = pdf.w / 6
    row_height = pdf.font_size * 2
    #
    pdf.add_page()

    def print_classification(image_filename_head, cl, y):
        image_filename_head = image_filename_head + '_' + y + '.png'
        filename = root_py + '\\reports\\plots\\' + image_filename_head.lower()
        #
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="Parameter " + y + ".", ln=1, align="C")
        pdf.set_font("Arial", size=9)
        pdf.cell(0, 10, txt="Most importance feature is " + cl['most_importance']['feature_name'] +
                            " with importance equal " + str(
            round((cl['most_importance']['feature_importance'] * 100), 2)) + "%",
                 ln=1)
        #
        pdf.cell(0, 10, txt="Statistic of feature " + y,
                 ln=1)
        ks = cl['y_feature'].keys()
        for k in ks:
            if k != "feature_importance":
                pdf.cell(col_width, row_height * 1,
                         txt=k, border=1)
        pdf.ln(row_height * 1)
        for key, val in cl['y_feature'].items():
            if key != "feature_importance":
                pdf.cell(col_width, row_height * 1,
                         txt=str(val), border=1)

        pdf.ln(row_height * 1)
        #
        pdf.cell(0, 10, txt="Statistics of the significance of each of the parameters",
                 ln=1)
        x = cl['x_features'].keys()
        y = [val["feature_importance"] for _, val in cl['x_features'].items()]
        for key, val in cl['x_features'].items():
            pdf.cell(col_width, row_height * 1, txt="Feature : " + key, ln=1)
            ks = val.keys()
            for k in ks:
                pdf.cell(col_width, row_height * 1,
                         txt=k, border=1)
            pdf.ln(row_height * 1)
            for k, v in val.items():
                if k == "feature_importance":
                    pdf.cell(col_width, row_height * 1,
                             txt=str(round((v * 100), 2)) + "%", border=1)
                else:
                    pdf.cell(col_width, row_height * 1,
                             txt=str(round(v, 2)) + "%", border=1)
            pdf.ln(row_height * 1)
        # fig = plt.figure()
        # ax = fig.add_axes([0, 0, 1, 1])
        # ax.set_ylim([0.0, 100.0])
        # ax.bar(x, y)
        # fig.show()
        # fig.savefig(filename)
        # pdf.image(filename, x=10, y=8, w=100)
        pdf.add_page()

    #
    pdf.set_font("Arial", size=18)
    pdf.cell(0, 10, txt="Classification by world.", ln=1, align="C")
    print_classification('cl_world', cl_world_rank, 'Rank')
    print_classification('cl_world', cl_world_score, 'Score')
    #
    pdf.set_font("Arial", size=18)
    pdf.cell(0, 10, txt="Classification by regions.", ln=1, align="C")
    for region, _ in map_region_and_rows.items():
        if isinstance(region, str):
            pdf.set_font("Arial", size=16)
            pdf.cell(0, 10, txt="Statistic of " + region + " region.", ln=1, align="C")
            print_classification('cl_region_' + region, cl_region_rank[region], 'Rank')
            print_classification('cl_region_' + region, cl_region_score[region], 'Score')
    #
    pdf.set_font("Arial", size=18)
    pdf.cell(0, 10, txt="Classification by countries.", ln=1, align="C")
    for country, _ in map_country_and_rows.items():
        #
        if isinstance(country, str):
            pdf.set_font("Arial", size=16)
            pdf.cell(0, 10, txt="Statistic of " + country + " country.", ln=1, align="C")
            print_classification('cl_country_' + country, cl_country_rank[country], 'Rank')
            print_classification('cl_country_' + country, cl_country_score[country], 'Score')
    #
    pdf.set_font("Arial", size=18)
    pdf.cell(0, 10, txt="Clusterization.", ln=1, align="C")
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, txt="Centers of clusters.", ln=1, align="C")
    pdf.set_font("Arial", size=9)
    for cluster, _ in enumerate(kmeans_clustering["centers"]):
        pdf.cell(0, 10, txt="Cluster number " + str(cluster), ln=1)
        for x in predict.x_feature_names:
            pdf.cell(col_width, row_height * 1,
                     txt=x, border=1)
        pdf.ln(row_height * 1)
        for center, _ in enumerate(kmeans_clustering["centers"][cluster]):
            pdf.cell(col_width, row_height * 1,
                     txt=str(round(kmeans_clustering["centers"][cluster][center], 2)) + "%", border=1)
        pdf.ln(row_height * 1)
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, txt="Countries of clusters.", ln=1, align="C")
    pdf.set_font("Arial", size=9)
    for cluster, _ in enumerate(kmeans_clustering["countries"]):
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="Cluster number " + str(cluster), ln=1, align="C")
        pdf.set_font("Arial", size=9)
        for _, country in enumerate(kmeans_clustering["countries"][cluster]):
            pdf.cell(0, 10, txt=" - Country: " + country["country"] + " , with distance from center of cluster " + str(
                round(country["distance"], 2)), ln=1)
        pdf.add_page()
    #
    pdf.output(root_py + '\\reports\\report.pdf')
    return


def info_classification(scope, y_feature_name):
    response = dict()
    if scope == 'region':
        list_regions = list()
        cl_by_region = None
        if y_feature_name == 'rank':
            cl_by_region = cl_region_rank
        if y_feature_name == 'score':
            cl_by_region = cl_region_score
        for key, val in cl_by_region.items():
            region = dict()
            region['region'] = key
            region['most_importance'] = val['most_importance']
            region['x_features'] = val['x_features']
            region['y_feature'] = val['y_feature']
            list_regions.append(region)
        response['regions'] = list_regions
        return response
    if scope == 'country':
        list_countries = list()
        cl_by_country = None
        if y_feature_name == 'rank':
            cl_by_country = cl_country_rank
        if y_feature_name == 'score':
            cl_by_country = cl_country_score
        for key, val in cl_by_country.items():
            country = dict()
            country['country'] = key
            country['most_importance'] = val['most_importance']
            country['x_features'] = val['x_features']
            country['y_feature'] = val['y_feature']
            list_countries.append(country)
        response['countries'] = list_countries
        return response
    if scope == 'world':
        cl_by_world = None
        if y_feature_name == 'rank':
            cl_by_world = cl_world_rank
        if y_feature_name == 'score':
            cl_by_world = cl_world_score
        response['most_importance'] = cl_by_world['most_importance']
        response['x_features'] = cl_by_world['x_features']
        response['y_feature'] = cl_by_world['y_feature']
        return response
    return None


def predict_classification_region(region, x_predict_features, y_feature_name):
    cl_by_region = None
    if y_feature_name == 'rank':
        cl_by_region = cl_region_rank
        y_feature_name = 'Rank'
    if y_feature_name == 'score':
        cl_by_region = cl_region_score
        y_feature_name = 'Score'
    response = predict.classification_by_column(
        value_column=region,
        cl_by_column=cl_by_region,
        x_predict_values=x_predict_features,
        map_column_and_rows=map_region_and_rows,
        y_feature_name=y_feature_name,
        flag='region'
    )
    return response


def predict_classification_country(country, x_predict_features, y_feature_name):
    cl_by_country = None
    if y_feature_name == 'rank':
        cl_by_country = cl_country_rank
        y_feature_name = 'Rank'
    if y_feature_name == 'score':
        cl_by_country = cl_country_score
        y_feature_name = 'Score'
    response = predict.classification_by_column(
        value_column=country,
        cl_by_column=cl_by_country,
        x_predict_values=x_predict_features,
        map_column_and_rows=map_country_and_rows,
        y_feature_name=y_feature_name,
        flag='country'
    )
    return response


def predict_classification_world(x_predict_features, y_feature_name):
    cl_by_world = None
    if y_feature_name == 'rank':
        cl_by_world = cl_world_rank
        y_feature_name = 'Rank'
    if y_feature_name == 'score':
        cl_by_world = cl_world_score
        y_feature_name = 'Score'
    response = predict.classification_by_world(
        x_predict_values=x_predict_features,
        cl_by_world=cl_by_world,
        y_feature_name=y_feature_name,
        concat_data_frame=concat_data_frame
    )
    return response


def predict_kmeans(x_predict_features):
    response = predict.clustering_kmeans(
        kmeans_clustering=kmeans_clustering,
        x_predict_values=x_predict_features
    )
    return response


def predict_happiness_models(x_predict_features):
    response = predict.happiness_models(
        hp_models=hp_models,
        x_predict_values=x_predict_features
    )
    return response
