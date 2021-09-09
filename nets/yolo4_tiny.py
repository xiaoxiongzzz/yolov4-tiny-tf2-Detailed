import tensorflow as tf
from nets.CSPdarknet53_tiny import darknet_body

def yolo_body(inputs,num_anchors,num_classes,phi = 0):
    if phi >=4:
        raise AssertionError("Phi must be less than or equal to 3 (0,1,2,3).")
    # ---------------------------------------------------#
    #   生成CSPdarknet53_tiny的主干模型
    #   feat1的shape为26,26,256
    #   feat2的shape为13,13,512
    # ---------------------------------------------------#
    feat1, feat2 = darknet_body(inputs)
