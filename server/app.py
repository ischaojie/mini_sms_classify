from flask import Flask, jsonify, request

import server.models.sms_classify as clf

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return jsonify('sms classify api.')


# 垃圾邮件检测(英文版)
@app.route('/sms', methods=['GET', 'POST'])
def sms_classify():
    email = request.args.get('email', '')
    result = clf.predict(email)
    return jsonify(email=email, label=result)


if __name__ == '__main__':
    app.run(debug=True)
