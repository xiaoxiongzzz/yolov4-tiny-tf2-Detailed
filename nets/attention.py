from tensorflow.keras.layers import (Activation, Add, Concatenate, Conv1D, Conv2D, Dense,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda,
                          Reshape, multiply)
from tensorflow.keras import backend as K
import math
# se 通道注意力机制，给通道赋予不同权重，简单好用。假设输入的是一个 h * w * c 的 feature map,
# 首先对它进行一个 global average pooling ，由全局池化（池化大小为 h * w）的操作我们可以得到一个 1 * 1 * c 的 feature map ，
# 然后就是两个全连接层，第一个全连接层的神经元个数为 c/16（作者给的参数），这就是一个降维的方法，第二个全连接层又升维到了 C 个神经元个数，
# 这样做的好处是增加了更多的非线性处理过程，可以拟合通道之间复杂的相关性。然后再接一个sigmod层，
# 得到 1 * 1 * c 的 feature map，最后就是一个原始的h * w * c 和 1 * 1 * c 的 feature map 全乘的操作。
# 之所以是全乘不是矩阵相乘，那是因为这样可以得到不同通道重要性不一样的 feature map。（可以了解矩阵全乘和矩阵相乘的概念和计算方法）
def se_block(intput_feature, ratio=16, name = ""):# 神经元个数为c/16
    channel = K.int_shape(intput_feature)[-1]# 初始化对象

    se_feature = GlobalAveragePooling2D()(intput_feature)
    se_feature = Reshape((1,1,channel))(se_feature)# 拉平

    # 第一个全连接层
    se_feature = Dense(channel//ratio,
                       activation='relu',
                       kernel_initializer='he_normal',# 卷积核初始化
                       use_bias=False,
                       name ="se_block_one_"+str(name))(se_feature)

    # 第二个全连接层
    se_feature = Dense(channel,
                       kernel_initializer='he_normal',
                       use_bias=False,
                       name = "se_block_two_"+str(name))(se_feature)
    # 激活层sigmoid
    se_feature = Activation('sigmoid')(se_feature)
    # 矩阵的全乘
    se_feature = multiply([intput_feature,se_feature])
    return se_feature

def channel_attention(input_feature, ratio=8, name=""):
    channel = K.int_shape(input_feature)(-1) # 初始化

    avg_pool = GlobalAveragePooling2D()(input_feature) # 全局平均池化
    max_pool = GlobalAveragePooling2D()(input_feature) # 全局最大池化
    # 拉平
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    max_pool = Reshape((1, 1, channel))(max_pool)
    # 共享层
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',# 权重初始化 LeCun均匀分布初始化方法，参数由[-limit, limit]的区间中均匀采样获得，其中limit=sqrt(6 / fan_in), fin_in是权重向量的输入单元数（扇入）
                             use_bias=False,
                             bias_initializer='zeros',
                             name="channel_attention_shared_one_"+str(name)
                             )

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',# 权重初始化 LeCun均匀分布初始化方法，参数由[-limit, limit]的区间中均匀采样获得，其中limit=sqrt(6 / fan_in), fin_in是权重向量的输入单元数（扇入）
                             use_bias=False,
                             bias_initializer='zeros',
                             name="channel_attention_shared_two_"+str(name)
                             )

    avg_pool = shared_layer_one(avg_pool)
    max_pool = shared_layer_one(max_pool)

    avg_pool = shared_layer_two(avg_pool)
    max_pool = shared_layer_two(max_pool)
    # 将MLP输出的特征进行基于element-wise的加和操作，再经过sigmoid激活操作，生成最终的channel attention feature，
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    # 直接数字相乘和输入特征图
    return multiply([input_feature, cbam_feature])
    # 空间注意力机制
def spatial_attention(input_feature,name=""):
    kernel_size = 7  # 卷积核大小7or1

    cbam_feature = input_feature

    # 注意：这个不是全局平均池化和最大池化单个特整层，这个是基于通道的全局平均池化，也就是最后只生成一个特整层
    avg_pool = Lambda(lambda x:K.mean(x,axis=3,keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x:K.max(x,axis=3,keepdims=True))(cbam_feature)
    # 按通道叠加
    concat = Concatenate(axis=3)([avg_pool,max_pool])
    # filters 为此Conv2D的厚度，
    cbam_feature=Conv2D(filters=1,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='same',
                        kernel_initializer='he_normal',
                        use_bias=False,
                        name="spatial_attention_"+str(name))(concat)
    return multiply([input_feature,cbam_feature])
# cabm 通道注意力机制与空间注意力机制的结合。se的强化
def cbam_block(cbam_feature, ratio=8, name=""):
    cbam_feature = channel_attention(cbam_feature,ratio,name=name)
    cbam_feature = spatial_attention(cbam_feature, name=name)
    return cbam_feature

# eca 去掉了se的模块通道缩减，2020年论文，作者证明优化后的ECA追锂基酯效果效果好于SE
# 具体核心是将se注意力机制中的全连接层换成了效率较高的一维卷积
def eca_block(input_feature, b=1, gamma=2, name=""):
    channel = K.int_shape(input_feature)[-1]
    kernel_size = int(abs((math.log(channel,2)+b)/gamma)) # 论文中K的确定公式，一般使用取3。
    kernel_size = kernel_size if kernel_size % 2 else kernel_size+1 # 保证是单数

    avg_pool = GlobalAveragePooling2D()(input_feature)

    x = Reshape((-1, 1))(avg_pool)# 按列拉平
    # 一维卷积
    x = Conv1D(1, kernel_size=kernel_size, padding="same", name="eca_layer_" + str(name), use_bias=False, )(x)

    x =Activation('sigmoid')(x)
    x =Reshape((1,1,-1))(x)

    output = multiply([input_feature, x])
    return output
