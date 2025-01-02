# urls.py 路由文件

from .API.YyApiPss import YyApiPss
from .API.YyApiSeg import YyApiSeg
from .API.YyApiTess import YyApiTess
from .API.ZhmApi0 import ZhmApi0
from .API.ZhmApi1 import ZhmApi1
from .API.ZhmApi2 import ZhmApi2
from .API.ZhmApi3 import ZhmApi3
from .extension import api

# 异常图像检测算法
api.add_resource(ZhmApi0, '/api/dr/quality')
# 四期非四期诊断
api.add_resource(ZhmApi1, '/api/dr/4orNo4Detect')
# 基于视盘象限划分
api.add_resource(ZhmApi2, '/api/dr/disk')
# 病灶目标检测
api.add_resource(ZhmApi3, '/api/dr/bingzhao')


# tess豹纹状病灶
api.add_resource(YyApiTess, '/api/myopia/tess')
# pas后巩膜葡萄肿
api.add_resource(YyApiPss, '/api/myopia/pss')
# 其余病灶分割
api.add_resource(YyApiSeg, '/api/myopia/seg')
