# -----------------------#
#   对单张图片的预测功能复写  #
#      LXX复写任务的开始。  #
# -----------------------#
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from yolo import YOLO
import time
import os
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
    picture_save_path="./img"

    if mode =="predict":
        '''     @xiaoxiongzzz
               1、该代码我自己写了一个批量预测代码 mode为path predict，目标文件夹请自己进入代码修改，小白敲代码  
               代码并不完善，如果可以请给个小星星给我的github。
               
               2、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
               3、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
               4、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
               在原图上利用矩阵的方式进行截取。
               5、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
               比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
               @xiaoxiongzzz
               6、这里我写了一个可以判断所有图里检测到的某类别的个数，如果需要可以留言我修改代码。
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

    if mode == "path predict":
    # 批量预测
        while True:
            path = input('Input your path')
            image_ids = os.listdir(path)
            class_nums = 0
            if not os.path.exists(picture_save_path):
                os.makedirs(picture_save_path)
            for image_id in image_ids:
                image_path =picture_save_path + image_id
                image = Image.open(image_path)
                # 开启后在之后计算mAP可以可视化
                # image.save("./input/images-optional/"+image_id+".jpg")
                result,class_num = yolo.detect_image(image)
                class_nums=class_nums+class_num
                # image = Image.open(result)
                print(class_nums)
                img = cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(picture_save_path + image_id , img)
            break


    elif mode =="video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID'),# 该参数是MPEG-4编码类型，文件名后缀为.avi
            # #fourcc意为四字符代码（Four-Character Codes），顾名思义，该编码由四个字符组成,下面是VideoWriter_fourcc对象一些常用的参数，注意：字符顺序不能弄混
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # 相当于一个容器 将每一帧插入并保存
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

            fps = 0.0

            while (True):
                t1 = time.time()
                # 读取某一帧
                ref, frame = capture.read()
                # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转变成Image
                frame = Image.fromarray(np.uint8(frame))
                # 进行检测
                frame = np.array(yolo.detect_image(frame))
                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # fps的计算公式，时间的倒数
                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %.2f" % (fps))
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                if video_save_path != "":
                    out.write(frame)

                if c == 27:
                    capture.release()
                    break
            capture.release()
            out.release()
            cv2.destroyAllWindows()

        elif mode == "fps":
            test_interval = 100
            img = Image.open('img/street.jpg')
            tact_time = yolo.get_FPS(img, test_interval)
            print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
        else:
            raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")

