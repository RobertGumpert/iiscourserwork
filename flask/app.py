import os

from flask import Flask
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
    split_message = message.split(':')
    if split_message[0] == 'Авторская оценка, анализ':
        response = tgstrings.info_classification(split_message=split_message)
        return response
    if split_message[0] == 'Предсказать по авторской оценке':
        response = tgstrings.predict_classification(split_message=split_message)
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
