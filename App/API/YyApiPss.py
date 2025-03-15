import hashlib
import os

from flask import json, jsonify, Response, request
from flask_restful import Resource, fields, marshal_with, reqparse
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from PIL import Image
import io
from App.API import Result, make_response, cache, logging, auth, result_data
from App.predicts.YyPredictPss import YyPredictPss


def make_cache_key():
    json_data = request.get_json()
    if json_data is None:
        # 如果没有 JSON 数据，返回一个默认的缓存键或错误处理
        return "default_cache_key"
    sorted_json = json.dumps(json_data, sort_keys=True)
    # 使用 hashlib 创建一个基于 JSON 字符串内容的哈希值
    file_hash = hashlib.md5(sorted_json.encode()).hexdigest()
    # 文件名作为缓存键的一部分（如果需要）
    return f"{file_hash}_{request.path}"


class YyApiPss(Resource):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('imagePath', type=str, required=True, help="Image file Path is required")
        self.parser.add_argument('visitNumber', type=str, required=True, help="Visit Number is required")

    # 自定义缓存键函数，根据请求参数生成唯一的键

    @marshal_with(Result)
    # @auth.login_required
    # @cache.cached(timeout=60, key_prefix=make_cache_key)
    def post(self):
        print("后巩膜葡萄肿检测")
        args = self.parser.parse_args()
        self.logger.debug(args)
        image_path = args['imagePath']
        # 打开图片文件
        try:
            # 使用模型进行预测
            result = ""
            res_pss = YyPredictPss.predict_pss(image_path)
            if res_pss == "pss":
                result = "后巩膜葡萄肿"
            elif res_pss == "normal":
                result = "无后巩膜葡萄肿"
            # 返回结果
            return make_response(result_data(result, ""), 200, 'OK')
        except IOError as e:
            self.logger.error(f"Error opening or processing the image file: {e}")
            return make_response({"message": "Error processing image"}, 500)

    """
        'form'：表单数据
        'args'：查询字符串
        'headers'：请求头
        'cookies'：cookies
        'files'：文件上传
        'json'：JSON 数据
    """
