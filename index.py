import cv2
from tracker import Tracker
from ultralytics import YOLO
from util import (
        draw_counter,
        draw_label,
        # update_counter
    )
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
RED = (0, 0, 255)

video = cv2.VideoCapture("Untitled.mp4")

model = YOLO("./models/yolov8s-300epoch.pt")
model.fuse()

counter = {"car": 0, "truck": 0, "motorbike": 0, "bus": 0}

tracker = Tracker()

line_point1 = (125, 480)
line_point2 = (738, 705)
offset = 6

# {0: 'bus', 1: 'car', 2: 'motorbike', 3: 'truck'}
classes_dict = model.model.names  # type: ignore

while video.isOpened():
    ret, frame = video.read()
    if type(frame) is None:
        break

    cv2.line(frame, line_point1, line_point2, color=RED, thickness=3)

    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    if ret and results[0].boxes is not None:

        bounding_box = results[0].boxes.xyxy.tolist()
        classes_idx = results[0].boxes.cls.tolist()
        confidence_score = results[0].boxes.conf
        tracked_bounding_box = tracker.update(bounding_box)

        draw_counter(frame, counter)

        for bbox, cls, conf in zip(tracked_bounding_box, classes_idx, confidence_score):

            x1, y1, x2, y2, object_id = [int(value) for value in bbox]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color=GREEN, thickness=1)

            draw_label(frame, classes_dict, cls, x1, y1, object_id)

#            update_counter(cls, counter) # TO DO: ADD TRACKER TO PREVENT DUPLICATE DETECTION

        cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
