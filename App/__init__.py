from flask import Flask
from .extension import init_extensions
from .urls import *


def create_app():
    # 配置app,静态文件，模版文件目录
    app = Flask(__name__)
    # 注册插件
    init_extensions(app)
    return app
