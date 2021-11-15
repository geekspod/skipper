import cv2
import numpy as np


class Detector:
    def __init__(self, video):
        self.detected_objects = None
        self.video = video
        self.detector = cv2.HOGDescriptor()
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self):
        boxes, weights = self.detector.detectMultiScale(self.video.frame, winStride=(8, 8))
        self.detected_objects = np.array(boxes)
        return self.detected_objects

    def draw_detected_objects(self, frame):
        for i, (x, y, w, h) in enumerate(self.detected_objects):
            cv2.putText(frame, "Person " + str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
