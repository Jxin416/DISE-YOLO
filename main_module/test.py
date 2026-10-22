#"runs/train/exp45/weights/best.pt"(DISE-YOLO)

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp46/weights/best.pt') # 自己训练结束后的模型权重
    model.val(data='D:/YOLO/ultralytics-v8-3-161/data.yaml',
              split='test',
              imgsz=640,
              batch=4,
              workers=0,
              device='',
              optimizer='SGD',
              save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='exp',
              )