import os

from flask import Flask, send_file
from flask import request
from flask import jsonify
import requests

import service
import tgstrings

service.init()


def get_url(method):
    tg_api_domain = str(service.tg_configs['TELEGRAM']['url']).replace('"', '')
    token = str(service.tg_configs['TELEGRAM']['token']).replace('"', '')
    if not tg_api_domain or not token:
        print('error : some configs data is null')
        os._exit(1)
    return f'{tg_api_domain}{token}/{method}'


def tg_follow_updates_web_hook():
    domain = str(service.tg_configs['TELEGRAM']['domain']).replace('"', '')
    if not domain:
        print('error : some configs data is null')
        os._exit(1)
    tg_api_url = get_url('setWebhook')
    follower_domain_url = f'{domain}/tg/webhook/update'
    response = requests.post(tg_api_url, data={
        'url': follower_domain_url
    })
    json = response.json()
    if response.status_code != 200:
        print(json)
        os._exit(1)
    print(json)


tg_follow_updates_web_hook()

# -FLASK----------------------------------------------------------------------------------------------------------------

app = Flask(__name__)


# -TgBot----------------------------------------------------------------------------------------------------------------


def parse_message(message):
    response = {'error': 'bad params'}
    if message == "/report":
        domain = str(service.tg_configs['TELEGRAM']['domain']).replace('"', '')
        return f'{domain}/report/pdf'
    if message == "/start" or message == "/help":
        return "Что умеет чат бот?\n" \
               "1) Рассказывать тебе что важнее всего для счастья.\n" \
               "   По уровню:\n" \
               "   - в регионе мира\n" \
               "   - в стране\n" \
               "   - в мире\n" \
               "   Счастья измеряется по рангу страны в мире и по оценке жителей\n" \
               "   Если ты хочешь узнать, то вводи подобные команды:\n" \
               "   - Анализ: по регионам-Western Europe, оценка жителями\n" \
               "   - Анализ: по регионам-Western Europe, ранг\n" \
               "   - Анализ: по странам-Russia, оценка жителями\n" \
               "   - Анализ: по странам-Russia, ранг\n" \
               "   - Анализ: по миру, оценка жителями\n" \
               "   - Анализ: по миру, ранг\n" \
               "2) Автор долго изучал как живут люди в странах,\n" \
               "   поэтому ты можешь угадать в какой стране тебе\n" \
               "   лучше жить по мнению автора.\n" \
               "   Для этого введи сначала 'Предсказать по авторской оценке:',\n" \
               "   укажи уровень:\n" \
               "   - в регионе мира\n" \
               "   - в мире\n" \
               "   затем укажи насколько для тебя важные такие параметры (0-100):\n" \
               "   - ВВП: GDP\n" \
               "   - Семья: Family\n" \
               "   - Здравооранение: Health\n" \
               "   - Свобода: Freedom\n" \
               "   - Доверие к государ.: Corruption\n" \
               "   В итоге должен получить примерно такую комманду:\n" \
               "   - Предсказать по авторской оценке: по регионам-Western Europe, оценка жителями,\n" \
               "     GDP=0;Family=0;Health=0;Freedom=0;Corruption=0\n" \
               "   С этим пуунктом еще не все!\n" \
               "   Указав уровень 'по странам-', и указав страну, ты можешь\n" \
               "   угадать, станет ли жить страна счастливее,\n" \
               "   если ты указал такие параметры\n" \
               "   - Предсказать по авторской оценке: по странам-Russia, ранг,\n" \
               "     GDP=0;Family=0;Health=0;Freedom=0;Corruption=0\n" \
               "3)Плевать на статистику счастья?\n" \
               "   Вводи и не прогадаешь в какой стране тебе жить:\n" \
               "   - Мне повезет: GDP=0;Family=0;Health=0;Freedom=0;Corruption=0\n" \
               "4)Плевать даже на страну? Просто хочешь узнать будешь ли счастлив?\n" \
               "   Вводи:\n" \
               "   - Я буду счастлив: GDP=0;Family=0;Health=0;Freedom=0;Corruption=0\n"

    split_message = message.split(':')
    if split_message[0] == 'Анализ':
        response = tgstrings.info_classification(split_message=split_message)
        return response
    if split_message[0] == 'Предсказать по авторской оценке':
        response = tgstrings.predict_classification(split_message=split_message)
        return response
    if split_message[0] == 'Мне повезет':
        response = tgstrings.predict_kmeans(split_message=split_message)
        return response
    # Я буду счастлив
    if split_message[0] == 'Я буду счастлив':
        response = tgstrings.predict_happiness_models(split_message=split_message)
        return response
    return response


@app.route('/tg/webhook/update', methods=["GET", "POST"])
def tg_get_web_hook_update():
    tg_response_url = get_url('sendMessage')
    response_data = dict(
        chat_id=request.json['message']['chat']['id'],
        text=str(parse_message(request.json['message']['text'])),
    )
    response = requests.post(tg_response_url, data=response_data)
    if response.status_code != 200:
        print('Error response to client ', request.json['message']['chat']['id'], ' with status :',
              response.status_code, ' and info :', response.json())
        return {
            "ok": False
        }
    print('OK response to client ', request.json['message']['chat']['id'], ' with status :', response.status_code)
    return {
        "ok": True
    }


# -API------------------------------------------------------------------------------------------------------------------

@app.route('/report/pdf', methods=['GET'])
def get_pdf_report():
    filename = service.root_py + "\\reports\\report.pdf"
    return send_file(filename, mimetype='application/pdf')


@app.route('/info/classification', methods=['GET'])
def api_info_classification():
    scope = request.args.get('scope')
    y_feature_name = request.args.get('y')
    response = service.info_classification(scope=scope, y_feature_name=y_feature_name)
    if response is None:
        return jsonify({'error': 'bad params'}), 400
    return jsonify(response), 200


@app.route('/predict/classification', methods=['POST'])
def api_predict_classification():
    scope = request.args.get('scope')
    y_feature_name = request.args.get('y')
    x_predict_features = request.json
    if scope == 'world':
        response = service.predict_classification_world(
            x_predict_features=x_predict_features,
            y_feature_name=y_feature_name
        )
        return jsonify(response), 200
    if scope == 'region':
        region = request.args.get('region')
        response = service.predict_classification_region(
            region=region,
            x_predict_features=x_predict_features,
            y_feature_name=y_feature_name
        )
        return jsonify(response), 200
    if scope == 'country':
        country = request.args.get('country')
        response = service.predict_classification_country(
            country=country,
            x_predict_features=x_predict_features,
            y_feature_name=y_feature_name
        )
        return jsonify(response), 200
    return jsonify({'error': 'bad params'}), 400


if __name__ == '__main__':
    app.run()
