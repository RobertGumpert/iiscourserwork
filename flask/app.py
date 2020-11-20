from flask import Flask
from flask import request
from flask import jsonify

import service

service.init()

# -FLASK----------------------------------------------------------------------------------------------------------------

app = Flask(__name__)


# -TgBot----------------------------------------------------------------------------------------------------------------


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
