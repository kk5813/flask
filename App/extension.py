# flask 插件模块
from flask_caching import Cache
from flask_restful import Api
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from flasgger import Swagger
from werkzeug.security import generate_password_hash, check_password_hash

# 基本身份验证
auth = HTTPBasicAuth()
users = {
    "minAd": generate_password_hash("aiGao_kk_zc"),
    "common": generate_password_hash("adgQet2780dgJkl")
}


@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username


# 缓存设置
cache = Cache(config={
    # 'CACHE_TYPE': 'simple',
    'CACHE_TYPE': 'RedisCache',
    'CACHE_REDIS_HOST': 'localhost',
    'CACHE_REDIS_PORT': 6379,
    'CACHE_REDIS_DB': 0,
    'CACHE_DEFAULT_TIMEOUT': 300,  # 300 seconds = 5 minutes
})

# Api设置
api = Api()

# 跨域设置
cors = CORS(resources=r'/api/*', supports_credentials=True)

# Flasgger
swagger = Swagger()


def init_extensions(app):
    # 注册扩展
    cache.init_app(app)
    api.init_app(app)
    cors.init_app(app)
    swagger.init_app(app)
