# -----------------------#
#   对单张图片的预测功能复写  #
#      LXX复写任务的开始。  #
# -----------------------#
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from yolo import YOLO
# 返回主机运行时可见的物理设备列表。
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
# 设置是否应为PhysicalDevice.第一块GPU，并且启用内存增长
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ =="__main__":
    # 初始化 加载模型和各项参数
    yolo = YOLO()
    # -------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测
    #   'video'表示视频检测
    #   'fps'表示测试fps
    # -------------------------------------------------------------------------#
    mode = "predict"
    # -------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出才会完成完整的保存步骤，不可直接结束程序。
    # -------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    if mode =="predict":
        '''
               1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
               具体流程可以参考get_dr_txt.py，在get_dr_txt.py即实现了遍历还实现了目标信息的保存。
               2、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
               3、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
               4、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
               在原图上利用矩阵的方式进行截取。
               5、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
               比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
               '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()