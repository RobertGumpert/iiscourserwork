import service


# args - GDP=0;Family=0;Health=0;Freedom=0;Corruption=0
def get_x_predict_features(args):
    split_features = args.split(';')
    x_predict_features = dict()
    for item in split_features:
        split_item = item.split('=')
        x_predict_features[split_item[0].replace(" ", "")] = float(split_item[1].replace(" ", ""))
    return x_predict_features


# Команды:
#   Классификация, анализ:
#       Авторская оценка, анализ: по регионам-Western Europe, оценка жителями -> service.info_classification
#       Авторская оценка, анализ: по регионам-Western Europe, ранг -> service.info_classification
#       Авторская оценка, анализ: по миру, оценка жителями -> service.info_classification
#       Авторская оценка, анализ: по миру, ранг -> service.info_classification
#       Авторская оценка, анализ: по странам-Russia, оценка жителями -> service.info_classification
#       Авторская оценка, анализ: по странам-Russia, ранг -> service.info_classification
#
def info_classification(split_message):
    split_args = split_message[1].split(',')
    y_feature_name = ''
    if 'оценка жителями' in split_args[1]:
        y_feature_name = 'score'
    if 'ранг' in split_args[1]:
        y_feature_name = 'rank'
    if 'по регионам' in split_args[0]:
        scope = 'region'
        region = split_args[0].split('-')[1]
        info = service.info_classification(scope=scope, y_feature_name=y_feature_name)
        info = [x for x in info['regions'] if x['region'] == region]
        response = f'{info[0]["region"]}:\nНаиболее значимо для счастья {info[0]["most_importance"]["feature_name"]},' + \
                   f'со значимостью {str(int(info[0]["most_importance"]["feature_importance"] * 100))}% \nОстальные параметры:\n'
        other = ''
        for k, v in info[0]["x_features"].items():
            other += f'Параметр {k} со значимостью {str(int(v["feature_importance"] * 100))}%\n'
        response += other
        return response
    if 'по миру' in split_args[0]:
        scope = 'world'
        info = service.info_classification(scope=scope, y_feature_name=y_feature_name)
        response = f'По миру:\nНаиболее значимо для счастья {info["most_importance"]["feature_name"]},' + \
                   f'со значимостью {str(int(info["most_importance"]["feature_importance"] * 100))}% \nОстальные параметры:\n'
        other = ''
        for k, v in info["x_features"].items():
            other += f'Параметр {k} со значимостью {str(int(v["feature_importance"] * 100))}%\n'
        response += other
        return response
    if 'по странам' in split_args[0]:
        scope = 'country'
        country = split_args[0].split('-')[1]
        info = service.info_classification(scope=scope, y_feature_name=y_feature_name)
        info = [x for x in info['countries'] if x['country'] == country]
        response = f'{info[0]["country"]}:\nНаиболее значимо для счастья {info[0]["most_importance"]["feature_name"]},' + \
                   f'со значимостью {str(int(info[0]["most_importance"]["feature_importance"] * 100))}% \nОстальные параметры:\n'
        other = ''
        for k, v in info[0]["x_features"].items():
            other += f'Параметр {k} со значимостью {str(int(v["feature_importance"] * 100))}%\n'
        response += other
        return response
    return {'error': 'bad params'}


# Команды:
#   Классификация, предсказать:
#       Предсказать по авторской оценке: по регионам-Western Europe, ранг,
#                                                                   GDP=0;Family=0;Health=0;Freedom=0;Corruption=0
#       Предсказать по авторской оценке по регионам-Western Europe, оценка жителями,
#                                                                   GDP=0;Family=0;Health=0;Freedom=0;Corruption=0
#       Предсказать по авторской оценке: по миру, ранг,
#                                                                   GDP=0;Family=0;Health=0;Freedom=0;Corruption=0
#       Предсказать по авторской оценке: по миру, оценка жителями,
#                                                                   GDP=0;Family=0;Health=0;Freedom=0;Corruption=0
#       Предсказать по авторской оценке: по странам-Russia, оценка жителями,
#                                                                   GDP=0;Family=0;Health=0;Freedom=0;Corruption=0
#       Предсказать по авторской оценке: по странам-Russia, ранг,
#                                                                   GDP=0;Family=0;Health=0;Freedom=0;Corruption=0
#
def predict_classification(split_message):
    split_args = split_message[1].split(',')
    x_predict_features = get_x_predict_features(split_args[2])
    y_feature_name = ''
    if 'оценка жителями' in split_args[1]:
        y_feature_name = 'score'
    if 'ранг' in split_args[1]:
        y_feature_name = 'rank'
    if 'по регионам' in split_args[0]:
        scope = 'region'
        region = split_args[0].split('-')[1]
        info = service.predict_classification_region(
            region=region,
            x_predict_features=x_predict_features,
            y_feature_name=y_feature_name
        )
        response = ''
        if info['predict_class'] == info['classes'][0]:
            response += f'\nВам подойдёт : {info["predict"][-1]["Country"]}\n'
        if info['predict_class'] == info['classes'][-1]:
            response += f'\nВам подойдёт : {info["predict"][0]["Country"]}\n'
        response += f'{region}: {info["class"]}' + \
                    f'\nНаиболее значимо для счастья {info["most_importance"]["feature_name"]},' + \
                    f'со значимостью {str(int(info["most_importance"]["feature_importance"] * 100))}% \nОстальные параметры:'
        other = ''
        for k, v in info["statistics"].items():
            other += f'\n*\nПараметр {k} со значимостью {str(int(v["feature_importance"] * 100))}%\n' + \
                     f'Маск: {str(int(v["feature_max"]))}%, Мин: {str(int(v["feature_min"]))}%, Изменяется на: {str(int(v["feature_dispersion"]))}%'
        response += other
        return response
    if 'по миру' in split_args[0]:
        scope = 'world'
        info = service.predict_classification_world(
            x_predict_features=x_predict_features,
            y_feature_name=y_feature_name
        )
        response = ''
        if info['predict_class'] == info['classes'][0]:
            response += f'\nВам подойдёт : {info["predict"][-1]["Country"]}\n'
        if info['predict_class'] == info['classes'][1]:
            response += f'\nВам подойдёт : {info["predict"][int(len(info["predict"]) / 2)]["Country"]}\n'
        if info['predict_class'] == info['classes'][2]:
            response += f'\nВам подойдёт : {info["predict"][0]["Country"]}\n'
        response += f'По миру: {info["class"]}' + \
                    f'\nНаиболее значимо для счастья {info["most_importance"]["feature_name"]},' + \
                    f'со значимостью {str(int(info["most_importance"]["feature_importance"] * 100))}% \nОстальные параметры:'
        other = ''
        for k, v in info["statistics"].items():
            other += f'\n*\nПараметр {k} со значимостью {str(int(v["feature_importance"] * 100))}%\n' + \
                     f'Маск: {str(int(v["feature_max"]))}%, Мин: {str(int(v["feature_min"]))}%, Изменяется на: {str(int(v["feature_dispersion"]))}%'
        response += other
        return response
    if 'по странам' in split_args[0]:
        scope = 'country'
        country = split_args[0].split('-')[1]
        info = service.predict_classification_country(
            country=country,
            x_predict_features=x_predict_features,
            y_feature_name=y_feature_name
        )
        response = ''
        response += f'Страна {country} станет: {info["class"]}' + \
                    f'\nНаиболее значимо для счастья {info["most_importance"]["feature_name"]},' + \
                    f'со значимостью {str(int(info["most_importance"]["feature_importance"] * 100))}% \nОстальные параметры:'
        other = ''
        for k, v in info["statistics"].items():
            other += f'\n*\nПараметр {k} со значимостью {str(int(v["feature_importance"] * 100))}%\n' + \
                     f'Маск: {str(int(v["feature_max"]))}%, Мин: {str(int(v["feature_min"]))}%, Изменяется на: {str(int(v["feature_dispersion"]))}%'
        response += other
        return response
    return {'error': 'bad params'}
