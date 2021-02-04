"""
git clone https://github.com/ultralytics/yolov3
cd yolov3
python3 weight_change.py
# -------------------------------------------------
git clone https://github.com/ultralytics/yolov5
cd yolov5
python3 weight_change.py
"""

from models.yolo import Model
import torch

cfg = "./models/yolov3.yaml"
weights = "./weights/yolov3.pt" # 原始的权重文件保存了模型参数和模型结构
m = Model(cfg)
ckpt = torch.load(weights, map_location="cpu")
# model.load_state_dict(ckpt['model'])
ckpt["model"] = {k: v for k, v in ckpt["model"].state_dict().items() if m.state_dict()[k].numel() == v.numel()}
m.load_state_dict(ckpt["model"], strict=False)
print("load weights... successful")

# 重新保存权重（只保存权重，不保存模型）
torch.save({"model":m.state_dict()},"yolov3.pth")