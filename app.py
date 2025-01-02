from flask import jsonify

from App import create_app

app = create_app()


# 全局异常处理器
@app.errorhandler(Exception)
def global_exception_handler(e):
    return jsonify({'code': 500, 'data': {'error': str(e)}, 'msg': 'xxx'}), 500


if __name__ == '__main__':
    print('Starting')
    app.run(port=4091, host="0.0.0.0", debug=True)
