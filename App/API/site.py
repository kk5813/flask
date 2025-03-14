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
from App.model.wjl_site_model import SitePredict
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
        print("眼別识别开始")
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
            print(image_path)
            # 遍历所有图像路径并进行预测
            result_datas = []
            left_eye_found = False
            right_eye_found = False
            for path in image_path:
                eye = SitePredict(path)  # 假设 SitePredict 返回 0（左眼）或 1（右眼）

                if eye == 0 and not left_eye_found:  # 找到左眼且尚未记录
                    result_datas.append(result_data("左眼", path))
                    left_eye_found = True
                elif eye == 1 and not right_eye_found:  # 找到右眼且尚未记录
                    result_datas.append(result_data("右眼", path))
                    right_eye_found = True

                # 如果左右眼都找到了，提前退出循环
                if left_eye_found and right_eye_found:
                    break
            return make_response(result_datas, 200, 'OK')
        else:
            eye = SitePredict(image_path)  # 假设 SitePredict 返回 0 或 1
            if eye == 0:
                result = "左眼"
            else:
                result = "右眼"
            return make_response(result_data(result, image_path), 200, 'OK')

