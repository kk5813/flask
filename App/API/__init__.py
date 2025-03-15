# API 接口文件
import os

from flask_restful import fields
import logging.config
from App.extension import cache, auth

# 统一返回结构
Result = {
    'code': fields.Integer,
    'data': fields.Raw,
    'msg': fields.String
}

# save_path = os.path.join("/", "opt", "resources", "images", "diagnose/")
save_path = r"E:/Download/project/diagnose"

# 返回数据
def make_response(data, code, msg):
    return {'code': code, 'data': data, 'msg': msg}, code

print("test_init_response")

def result_data(resultInfo, url):
    return {'resultInfo': resultInfo, 'url': url}


""" asctime：日志生成的时间 , name：记录器的名称, levelname：日志的级别（如 DEBUG, INFO, ERROR） ,message：日志的实际内容"""
# 配置日志
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "info_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "simple",
                "filename": "info.log",
                "maxBytes": 10485760,
                "backupCount": 50,
                "encoding": "utf8",
            },
            "error_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "simple",
                "filename": "errors.log",
                "maxBytes": 10485760,
                "backupCount": 20,
                "encoding": "utf8",
            },
            "debug_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "filename": "debug.log",
                "maxBytes": 10485760,
                "backupCount": 50,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "my_module": {"level": "ERROR", "handlers": ["console"], "propagate": "no"}
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["error_file_handler", "debug_file_handler"],
        },
    }
)
