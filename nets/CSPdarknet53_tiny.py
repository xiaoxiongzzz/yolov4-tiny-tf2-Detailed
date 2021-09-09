import tensorflow as tf
from tensorflow.keras.initializers import  RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (BatchNormalization, Concatenate,
                                     Conv2D, Lambda, LeakyReLU,
                                    MaxPooling2D, ZeroPadding2D)
from functools import wraps
from utils.utils import compose
def darknet_body(x):
    # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
    # 416,416,3 -> 208,208,32 -> 104,104,64
    # 上边和左边各填充一行0，为了卷积能拿到更多信息
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    print("ZeroPadding2D=",x.shape)
    x = DarknetConv2D_BN_Leaky(32, (3,3), strides=(2,2))(x)
    print("DarknetConv2D_BN_Leaky32=",x.shape)
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    print("ZeroPadding2D=",x.shape)
    x = DarknetConv2D_BN_Leaky(64, (3, 3), strides=(2, 2))(x)
    print("DarknetConv2D_BN_Leaky64=",x.shape)
    # 104,104,64 ->52,52,128
    x, _ = resblock_body(x,num_filters = 64)
    # 52,52,128 -> 26,26,256
    x, _ = resblock_body(x, num_filters=128)
    # 26,26,256 -> x为13,13,512
    #           -> feat1为26,26,256
    x, feat1 = resblock_body(x, num_filters=256)
    # 13,13,512 -> 13,13,512
    x = DarknetConv2D_BN_Leaky(512, (3, 3))(x)

    feat2 = x
    return feat1, feat2

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args,**kwargs):
    no_bias_kwargs ={'use_bias':False}# 创建一个偏置字典，默认为False
    no_bias_kwargs.update(kwargs)# 更新

    return compose(
        DarknetConv2D(*args,**no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#--------------------------------------------------#
#   单次卷积DarknetConv2D
#   如果步长为2则自己设定padding方式。
#   测试中发现没有l2正则化效果更好，所以去掉了l2正则化
#--------------------------------------------------#
@wraps(Conv2D)  # 难点：Python装饰器（decorator）在实现的时候，被装饰后的函数其实已经是另外一个函数了（函数名等函数属性会发生改变），为了不影响，Python的functools包中提供了一个叫wraps的decorator来消除这样的副作用。写一个decorator的时候，最好在实现之前加上functools的wrap，它能保留原有函数的名称和docstring。
def DarknetConv2D(*args, **kwargs):
    #darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)} #  L2正则化 避免过拟合，惩罚损失函数。实际效果不太好，
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02)}# 标准差为0.02的随机正态分布初始化
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same' # 步长为2，2为valid,否则为same。
    darknet_conv_kwargs.update(kwargs)# update
    return Conv2D(*args, **darknet_conv_kwargs)#相当于对Conv2D进行了修改。但是为了不影响，所以加了wraps。

#---------------------------------------------------#
#   CSPdarknet_tiny的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
#---------------------------------------------------#
def resblock_body(x,num_filters):#filters 为输出卷积滤波器，也就是输出维度。
    # 利用一个3x3卷积进行特征整合，维度不变。
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3))(x)
    # 引出一个大的残差边route
    route = x

    # 对特征层的通道进行分割，取第二部分作为主干部分。
    x = Lambda(route_group, arguments={'groups':2, 'group_id':1})(x)
    print("resblock_body=", x.shape)

    # 对主干部分进行3x3卷积   通道数减半
    x = DarknetConv2D_BN_Leaky(int(num_filters / 2), (3, 3))(x)
    # 引出一个小的残差边route_1
    route_1 = x
    # 对第主干部分进行3x3卷积   通道数再减半
    x = DarknetConv2D_BN_Leaky(int(num_filters / 2), (3, 3))(x)
    # 主干部分与残差部分进行相接
    x = Concatenate()([x, route_1])
    # 对相接后的结果进行1x1卷积   通道数恢复输入通道数
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    feat = x
    x = Concatenate()([route, x]) # 大残差边与主干堆叠 通道数翻倍
    # 利用最大池化进行高和宽的压缩    图片大小减半
    x = MaxPooling2D(pool_size=[2, 2], )(x)
    # x为主干，feat为小残差边
    return x, feat





# 切割通道数代码
def route_group(input_layer,groups,group_id):
    # 对通道数进行分割，我们取用第二部分
    convs = tf.split(input_layer,num_or_size_splits=groups,axis=-1)
    return convs[group_id]
