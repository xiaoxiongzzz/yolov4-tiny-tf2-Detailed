# 复写代码第三步，最难啃的训练阶段，如何通过数据集训练出自己的权重一直是难点，
# 所以这一块需要慢慢写，慢慢读
import tensorflow as tf

import numpy as np
from tensorflow.keras.layers import Input, Lambda
from nets.yolo4_tiny import yolo_body
# GPU全打开
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #----------------------------------------------------#
    #   是否使用eager模式训练   一种简洁的训练展示模式
    #----------------------------------------------------#
    eager = False
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    annotation_path = '2007_train.txt'
    #------------------------------------------------------#
    #   训练后的模型保存的位置，保存在logs文件夹里面
    #------------------------------------------------------#
    log_dir = 'logs/'
    # ----------------------------------------------------#
    #   classes和anchor的路径，非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    # ----------------------------------------------------#
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    # ------------------------------------------------------#
    weights_path = 'model_data/yolov4_tiny_weights_coco.h5'
    # ------------------------------------------------------#
    #   训练用图片大小
    #   一般在416x416和608x608选择
    # ------------------------------------------------------#
    input_shape = (608, 608)
    # ------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    # ------------------------------------------------------#
    normalize = False
    #------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    #------------------------------------------------------#
    mosaic = False
    Cosine_scheduler = False
    label_smoothing = 0

    #------------------------------------------------------#
    #   在eager模式下是否进行正则化
    #------------------------------------------------------#
    regularization = True
    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    #------------------------------------------------------#
    #   一共有多少类和多少先验框
    #------------------------------------------------------#
    num_classes = len(class_names)
    num_anchors = len(anchors)

    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    print('Create YOLOv4-Tiny model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors // 2, num_classes) # 返回模型Model

    #-------------------------------------------#
    #   权值文件的下载请看README
    #-------------------------------------------#
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # y_true为13,13,3,85 难点，作者一行代码写出来较难理解  就是输出的维度大小。2个特征层一层3个anchors。
    # 26,26,3,85
    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], num_anchors//2, num_classes+5)) for l in range(2)]

    #------------------------------------------------------#
    #   在这个地方设置损失，将网络的输出结果传入loss函数
    #   把整个模型的输出作为loss
    #    难点：loss函数为训练时调整权值最重要的环节。
    #------------------------------------------------------#
    loss_input = [*model_body.output, *y_true]
    # 难点  loss的定义很难理解。
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                   'label_smoothing': label_smoothing, 'normalize': normalize})(loss_input)
