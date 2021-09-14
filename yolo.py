# 复写任务第二部yolo初始化
import tensorflow as tf
import numpy as np
import os
from nets.yolo4_tiny import yolo_body,yolo_eval
from tensorflow.keras.layers import Input, Lambda
import colorsys
from tensorflow.keras.models import Model

class YOLO(object):
    _defaults = {
        "model_path"       :'model_data/yolov4_tiny_weights_coco.h5',
        "anchors_path"     :'model_data/yolo_anchors.txt',
        "classes_path"     :'model_data/coco_classes.txt',
        # -------------------------------#
        #   所使用的注意力机制的类型
        #   phi = 0为不使用注意力机制
        #   phi = 1为SE
        #   phi = 2为CBAM
        #   phi = 3为ECA
        # -------------------------------#
        "phi": 0,
        "score":0.5,
        "iou":0.3,
        "max_boxes": 100,
        "model_image_size":(416,416),
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }
    # 类方法，方便以后获取默认值，类方法不需要实例。直接传给cls。
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo  kwargs为可变参数字典
    #---------------------------------------------------#
    def __init__(self,**kwargs):#kwargs为可变参数字典 这里为0
        self.__dict__.update(self._defaults)# 更新默认参数
        print(self._defaults)
        self.class_names = self._get_class()
        print(self.class_names)
        self.anchors = self._get_anchors()
        print(self.anchors)
        # 初始化模型
        self.generate()

    # 获取类别名
    def _get_class(self):
        class_path = os.path.expanduser(self.classes_path)#展开路径名，以防linux中的~
        with open(class_path) as f:
            class_names = f.readlines()
        class_names=[c.strip() for c in class_names]
        return class_names

    # 获取anchors
    def _get_anchors(self):
        anchors_path =os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline() # 只读一行
        anchors = [float(x) for x in anchors.split(',')] # 转换为列表并依次读取anchors

        return np.array(anchors).reshape(-1, 2)

    # 初始化模型
    def generate(self):
        model_path =os.path.expanduser(self.model_path)
        # 检查是否为h5结尾
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算先验框的数量和种类的数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #   载入模型
        # 返回输入和两个预测头P4 P5
        self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes, self.phi)
        print(self.yolo_model)
        self.yolo_model.load_weights(self.model_path) # 读取模型
        self.yolo_model.save_weights(self.model_path) # 保存模型
        print('{} model, anchors, and classes loaded.'.format(model_path))

    # 画框设置不同的颜色
        hsv_tuples = [(x/len(self.class_names),1.,1.)
                     for x in range(len(self.class_names))]# 设置类别个数的3通道颜色元组,hsv:色度 饱和度

        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x),hsv_tuples))# hsv转换成RGB的颜色格式 三个数值均小于1大于0.0
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))# 为颜色乘上255通道，以展示颜色。

    # 打乱颜色
        np.random.seed(10101) # 随机数种子
        np.random.shuffle(self.colors)# 打乱颜色
        np.random.seed(None)

        # ---------------------------------------------------------#
        #   在yolo_eval函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        # ---------------------------------------------------------#
        self.input_image_shape = Input([2, ],batch_size=1)# 模型输入维度为2，batch_size为1
        inputs = [*self.yolo_model.output, self.input_image_shape]# 定义输出维度与输入一致 这里读取了3个特征层，第一个为输出，2，3为输入
        # 对输出的feature map进行后处理。
        outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
                         arguments={'anchors': self.anchors, 'num_classes': len(self.class_names),
                                    'image_shape': self.model_image_size,
                                    'score_threshold': self.score, 'eager': True, 'max_boxes': self.max_boxes,
                                    'letterbox_image': self.letterbox_image})(inputs)
        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)

    # @tf.function
    # def get_pred(self, image_data, input_image_shape):
    #     out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
    #     return out_boxes, out_scores, out_classes


