from ultralytics import YOLO

import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel


class YOLO_custom(YOLO):
    def train(): #override ???
        pass
 
# Load the model.
#model = YOLO("yolov8n.yaml") #from scratch
model = YOLO('yolov8n.pt') #pretrained

# pred = model("data_ours/data")

# print(pred[0].boxes.cls)
# print(pred[0].boxes.xywhn)

#model.model.backbone

# for m in model.model.modules():
#    t = type(m)
#    print(t)
   #print(m)
print("____________")
BACKBONE = 9
for name, param in model.model.named_parameters():
    layer = int(name.split(".")[1])
    print(layer)
    if layer <= BACKBONE:
        param.requires_grad = False
    else:
        param.requires_grad = True
    print(name, param.size(), param.requires_grad)


#training
#model.train(data="coco128.yaml", epochs=1)  # train the model

# Training.
# results = model.train(
#    data='custom_data.yaml',
#    imgsz= 640,
#    epochs=1,
#    batch=8,
#    name='yolov8n_custom'
# )

def custom_train(model, epochs):
    for e in range(epochs):
        pass



# from : https://medium.com/cord-tech/yolov8-for-object-detection-explained-practical-example-23920f77f66a
# results = model.train(data="coco128.yaml", epochs=5) # train the model 
# results = model.val() # evaluate model performance on the validation data set 
# results = model("https://ultralytics.com/images/cat.jpg") # predict on an image
# success = YOLO("yolov8n.pt").export(format="onnx") # export a model to ONNX

#anotation:
# class_id center_x center_y width height #coords normalize [0,1]



#model.model :
# DetectionModel(
#   (model): Sequential(
#     (0): Conv(
#       (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#       (act): SiLU(inplace=True)
#     )
#     (1): Conv(
#       (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#       (act): SiLU(inplace=True)
#     )
#     (2): C2f(
#       (cv1): Conv(
#         (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (cv2): Conv(
#         (conv): Conv2d(48, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (m): ModuleList(
#         (0): Bottleneck(
#           (cv1): Conv(
#             (conv): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (cv2): Conv(
#             (conv): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#         )
#       )
#     )
#     (3): Conv(
#       (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#       (act): SiLU(inplace=True)
#     )
#     (4): C2f(
#       (cv1): Conv(
#         (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (cv2): Conv(
#         (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (m): ModuleList(
#         (0): Bottleneck(
#           (cv1): Conv(
#             (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (cv2): Conv(
#             (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#         )
#         (1): Bottleneck(
#           (cv1): Conv(
#             (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (cv2): Conv(
#             (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#         )
#       )
#     )
#     (5): Conv(
#       (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#       (act): SiLU(inplace=True)
#     )
#     (6): C2f(
#       (cv1): Conv(
#         (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (cv2): Conv(
#         (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (m): ModuleList(
#         (0): Bottleneck(
#           (cv1): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (cv2): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#         )
#         (1): Bottleneck(
#           (cv1): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (cv2): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#         )
#       )
#     )
#     (7): Conv(
#       (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#       (act): SiLU(inplace=True)
#     )
#     (8): C2f(
#       (cv1): Conv(
#         (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (cv2): Conv(
#         (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (m): ModuleList(
#         (0): Bottleneck(
#           (cv1): Conv(
#             (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (cv2): Conv(
#             (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#         )
#       )
#     )
#     (9): SPPF(
#       (cv1): Conv(
#         (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (cv2): Conv(
#         (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
#     )
#     (10): Upsample(scale_factor=2.0, mode=nearest)
#     (11): Concat()
#     (12): C2f(
#       (cv1): Conv(
#         (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (cv2): Conv(
#         (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (m): ModuleList(
#         (0): Bottleneck(
#           (cv1): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (cv2): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#         )
#       )
#     )
#     (13): Upsample(scale_factor=2.0, mode=nearest)
#     (14): Concat()
#     (15): C2f(
#       (cv1): Conv(
#         (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (cv2): Conv(
#         (conv): Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (m): ModuleList(
#         (0): Bottleneck(
#           (cv1): Conv(
#             (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (cv2): Conv(
#             (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#         )
#       )
#     )
#     (16): Conv(
#       (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#       (act): SiLU(inplace=True)
#     )
#     (17): Concat()
#     (18): C2f(
#       (cv1): Conv(
#         (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (cv2): Conv(
#         (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (m): ModuleList(
#         (0): Bottleneck(
#           (cv1): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (cv2): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#         )
#       )
#     )
#     (19): Conv(
#       (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#       (act): SiLU(inplace=True)
#     )
#     (20): Concat()
#     (21): C2f(
#       (cv1): Conv(
#         (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (cv2): Conv(
#         (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         (act): SiLU(inplace=True)
#       )
#       (m): ModuleList(
#         (0): Bottleneck(
#           (cv1): Conv(
#             (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (cv2): Conv(
#             (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#         )
#       )
#     )
#     (22): Detect(
#       (cv2): ModuleList(
#         (0): Sequential(
#           (0): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (1): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): Sequential(
#           (0): Conv(
#             (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (1): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): Sequential(
#           (0): Conv(
#             (conv): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (1): Conv(
#             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (cv3): ModuleList(
#         (0): Sequential(
#           (0): Conv(
#             (conv): Conv2d(64, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (1): Conv(
#             (conv): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (2): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): Sequential(
#           (0): Conv(
#             (conv): Conv2d(128, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (1): Conv(
#             (conv): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (2): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): Sequential(
#           (0): Conv(
#             (conv): Conv2d(256, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (1): Conv(
#             (conv): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#             (act): SiLU(inplace=True)
#           )
#           (2): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (dfl): DFL(
#         (conv): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       )
#     )
#   )
# )