from tensorflow.keras.layers import (Activation, Add, Concatenate, Conv1D, Conv2D, Dense,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda,
                          Reshape, multiply)
from tensorflow.keras import backend as K
# se 通道注意力机制，给通道赋予不同权重，简单好用。
def se_block(intput_feature, ratio=16, name = ""):
    channel = K.int_shape(intput_feature)[-1]# 初始化对象

    se_feature = GlobalAveragePooling2D()(intput_feature)
    se_feature = Reshape((1,1,channel))(se_feature)















# cabm 通道注意力机制与空间注意力机制的结合。se的强化
def cbam_block(cbam_feature, ratio=8, name=""):
# eca 去掉了se的模块通道缩减，2020年论文，作者证明优化后的ECA追锂基酯效果效果好于SE
def eca_block(input_feature, b=1, gamma=2, name=""):