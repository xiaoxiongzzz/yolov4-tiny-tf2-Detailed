# -----------------------#
#   对单张图片的预测功能复写  #
#      LXX复写任务的开始。  #
# -----------------------#
import tensorflow as tf
import cv2
import numpy as np
from yolo import YOLO
# 返回主机运行时可见的物理设备列表。
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
# 设置是否应为PhysicalDevice.第一块GPU，并且启用内存增长
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ =="__main__":
    yolo = YOLO()