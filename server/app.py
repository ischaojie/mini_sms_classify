from flask import Flask, jsonify, request
from flask_cors import CORS

import server.models.sms_classify as clf

app = Flask(__name__)
# 解决跨域问题
CORS(app)


@app.route('/', methods=['GET'])
def index():
    return jsonify('sms classify api, request </predict> get result.')


# 垃圾邮件检测(英文版)
@app.route('/predict', methods=['GET', 'POST'])
def sms_classify():
    email = request.args.get('email', '')
    if email == '':
        return jsonify(email=email, lable='')
    result = clf.predict(email)
    return jsonify(email=email, label=result)


if __name__ == '__main__':
    app.run(debug=True)
