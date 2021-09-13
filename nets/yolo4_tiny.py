import tensorflow as tf
from nets.CSPdarknet53_tiny import darknet_body
from nets.attention import se_block, cbam_block,eca_block
from nets.CSPdarknet53_tiny import DarknetConv2D_BN_Leaky,DarknetConv2D
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv2D,
                                     LeakyReLU, UpSampling2D)
from utils.utils import compose
from tensorflow.keras.models import Model


# 注意力机制列表
attention_block = [se_block, cbam_block, eca_block]
def yolo_body(inputs,num_anchors,num_classes,phi = 0):
    if phi >=4:
        raise AssertionError("Phi must be less than or equal to 3 (0,1,2,3).")
    # ---------------------------------------------------#
    #   生成CSPdarknet53_tiny的主干模型
    #   feat1的shape为26,26,256
    #   feat2的shape为13,13,512
    # ---------------------------------------------------#
    feat1, feat2 = darknet_body(inputs)# 返回2个特征层
    # ---------------------------------------------------#
    #   生成CSPdarknet53_tiny的主干模型
    #   feat1的shape为26,26,256
    #   feat2的shape为13,13,512
    # ---------------------------------------------------#
    if 1<=phi and phi<=3:
        feat1 = attention_block[phi-1](feat1, name="feat1")
        feat2 = attention_block[phi-1](feat2, name="feat2")

    # 从darknet_body出来的维度为13,13,512  先经过处理得到13,13,256,函数中256为输出通道数，后面为卷积核
    P5 = DarknetConv2D_BN_Leaky(256,(1,1))(feat2)
    # 13,13,256 ->13,13,128 -> 26,26,256
    P5_output = DarknetConv2D_BN_Leaky(512, (3,3))(P5)
    P5_output = DarknetConv2D(num_anchors*(num_classes+5), (1, 1))(P5_output)# 维度来源于论文公式

    # 13,13,256 -> 13,13,128 -> 26,26,128
    P5_upsample = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(P5)
    # 注意力机制的渗入
    if 1<=phi and phi<= 3:
        P5_upsample = attention_block[phi-1](P5_upsample, name="P5_upsample")

    # 26，26，256 + 26，26，128 -> 26，26，384 将P4和P5两个预测层合并
    P4 = Concatenate()([P5_upsample,feat1])

    # 26,26,384 -> 26,26,256 -> 26,26,255
    P4_output = DarknetConv2D_BN_Leaky(256, (3,3))(P4)
    P4_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(P4_output)
    # 2层的FPN结构，返回2个特征层
    # print(P4_output.shape,P5_output.shape)
    return Model(inputs, [P5_output, P4_output])