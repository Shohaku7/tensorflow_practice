from flask import Flask, request, jsonify

app = Flask(__name__) # 创建 Flask 实例

# 测试 URL
@app.route('/ping', methods=['GET', 'POST'])
def hello_world():
    return 'pong'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
    print("====")