import torch
from ultralytics import YOLO
from thop import profile
model = YOLO('D:/YOLO/ultralytics-v8-3-161/ultralytics/cfg/models/11/C3k2_iAFF-SELayerV1.yaml').model
input = torch.randn(1, 3, 640, 640)
flops, params = profile(model, inputs=(input,))
print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")