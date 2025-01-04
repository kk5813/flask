# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import os
import time
import uuid

import cv2
import numpy as np
from PIL import Image

from App.API import save_path
from App.model.yy_unet import Unet


class YyPredictSeg:
    def __init__(self, model_path):
        pass

    @staticmethod
    def predict_segment(img):
        # -------------------------------------------------------------------------  #
        #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
        # -------------------------------------------------------------------------#
        count = False
        name_classes = ["background", "CRA", "JSH", "od"]
        unet = Unet()
        image = Image.open(img)
        r_image = unet.detect_image(image, count=count, name_classes=name_classes)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = save_path + str(uuid.uuid1()) + ".jpg"
        r_image.save(path)

        # r_image.show()
        return path

        '''
        predict.py有几个注意点
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''


if __name__ == '__main__':
    img = YyPredictSeg.predict_segment(
        '../img/001752-20191015@100326-L3-S.jpg')
    print(img)
