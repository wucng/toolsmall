- https://github.com/ultralytics/yolov5
- https://github.com/ultralytics/yolov3

- https://github.com/AlexeyAB/darknet
- https://pjreddie.com/darknet/yolo/

---

- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137)  

---

- [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)
- [yolov3-spp.weights](https://pjreddie.com/media/files/yolov3-spp.weights)
- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) 416x416


# 查看模型结构

```python
from toolsmall.network.ultralytics.yolov3.models import Darknet
device = "cpu"
cfg = "./yolov3.cfg"
# Initialize model
model = Darknet(cfg).to(device) # stride=32,stride=16,stride=8
print(model)
#
torch.onnx.export(model,torch.rand([1,3,416,416],device=device),"model.onnx",verbose=True)
exit(0)

# 使用`netron` 打开 model.onnx 查看模型结构
```