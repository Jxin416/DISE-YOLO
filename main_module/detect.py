# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：detect.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""
#'D:/YOLO/ultralytics-v8-3-161/runs/train/exp/weights/best.pt'  46
#'D:/YOLO/ultralytics-v8-3-161/runs/train/exp40/weights/best.pt'  45

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'D:/YOLO/ultralytics-v8-3-161/runs/train/exp46/weights/best.pt')  
    model.predict(source=r'C:/Users/hp/Desktop/v11论文/图片/可视化实验/BJ_Bing_261(1).jpg',
                  save=True,
                  show=True,
                  )
