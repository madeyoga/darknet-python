import cv2
from darknet_yolov3 import Model
import json

img = cv2.imread('test_set/test2.jpg')

classes = ["left_data",
           "right_data",
           "id",
           "result_defeat",
           "result_victory",
           "score",
           "time",
           "report_button"]

model = Model('models/yolov3_mlbb_result_v3.weights',
              'models/yolov3_mlbb_result_v3.cfg',
              classes)

bbox = model.detect_bboxes(img)

print(json.dumps(bbox, sort_keys=True, indent=2))
