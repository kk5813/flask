from flask import json, jsonify, Response, request
from flask_restful import Resource, fields, marshal_with, reqparse
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from PIL import Image
import io
from App.API import Result, make_response, cache, logging, auth, result_data
from App.predicts.YyPredictTess import YyPredictTess



def make_cache_key():
    file_hash = secure_filename(request.files['image'].filename)  # 文件名作为缓存键的一部分
    return f"{file_hash}_{request.path}"


class YyApiTess(Resource):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('imagePath', type=str, required=True, help="Image file Path is required")

    # 自定义缓存键函数，根据请求参数生成唯一的键

    @marshal_with(Result)
    # @auth.login_required
    # @cache.cached(timeout=60, key_prefix=make_cache_key)
    def post(self):
        """
        第一个模型：大致描述模型做什么的，模型代号
        ---
        parameters:
          - in: body
            name: image
            type: files
            required: true
        responses:
          200:
            description: 成功
            schema:
                quality: 预测结果
                msg: OK
        """
        args = self.parser.parse_args()
        self.logger.debug(args)
        image_path = args['imagePath']
        result = ""
        tess = YyPredictTess.predict_tess(image_path)
        if tess == "tess":
            result = "豹纹状改变"
        elif tess == "normal":
            result = "无豹纹状改变"
        return make_response(result_data(result, ""), 200, 'OK')

    """
        'form'：表单数据
        'args'：查询字符串
        'headers'：请求头
        'cookies'：cookies
        'files'：文件上传
        'json'：JSON 数据
    """
