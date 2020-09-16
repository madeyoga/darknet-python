import numpy as np
import cv2

class Model:

    def __init__(self, weights_path, config_path, classes=[]):
        self.__weights_path = weights_path
        self.__config_path = config_path

        self.net = cv2.dnn.readNet(self.__weights_path,
                                   self.__config_path)

        self.classes = classes
        
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_bboxes(self, img, threshold=0.25):
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        entries = []
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
        for i in range(len(boxes)):
            if i in indexes:
                label = str(self.classes[class_ids[i]])
                entries.append((label, confidences[i], boxes[i]))

        return entries


