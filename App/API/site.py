import hashlib
import os
import random

from flask import json, jsonify, Response, request
from flask_restful import Resource, fields, marshal_with, reqparse
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from PIL import Image
import io
from App.API import Result, make_response, cache, logging, auth, result_data
from App.predicts.YyPredictPss import YyPredictPss
from App.util.pdfToJPG import process_pdf


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


# 入口
class site(Resource):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('imagePath', type=str, required=True, help="Image file Path is required")


    # 自定义缓存键函数，根据请求参数生成唯一的键

    @marshal_with(Result)
    # @auth.login_required
    # @cache.cached(timeout=60, key_prefix=make_cache_key)
    def post(self):
        args = self.parser.parse_args()
        self.logger.debug(args)
        image_path = args['imagePath']
        # 检查文件是否存在
        if not os.path.exists(image_path):
            return make_response("", 404, "File not found")
        # 判断文件是否是 PDF
        if image_path.lower().endswith('.pdf'):
            image_path = process_pdf(image_path, r'E:\python\flask_deploy\App\img\rendered',
                                     r'E:\python\flask_deploy\App\img\extracted', dpi=300)
            # 遍历所有图像路径并进行预测
            result_datas = []
            for path in image_path:
                result = SitePredict(path)  # 假设 SitePredict 返回 0 或 1
                if result in [0, 1]:
                    result_datas.append(result_data(result, path))
                    if len(result_datas) == 2:
                        break  # 如果找到结 果为 0 或 1，立即停止
            return make_response(result_datas, 200, 'OK')
        else:
            result = SitePredict()
            return make_response(result_data(result, image_path), 200, 'OK')


def SitePredict(image_path):
    return random.randint(0, 1)
