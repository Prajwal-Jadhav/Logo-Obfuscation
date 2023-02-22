import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path = './yolov5_logo.pt')

im = './pepsi_logo.jpg'

results = model(im)





