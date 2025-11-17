# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from datetime import datetime

import cvzone
import time
import json

import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

from counters import LineCrossCounter

#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

enable_gpu = True  # Set True if running with CUDA
model_file = "yolo11s.pt"  # Path to model file
show_fps = True  # If True, shows current FPS in top-left corner
show_conf = False  # Display or hide the confidence score
save_img = True  # Set True to save output video


conf = 0.3  # Min confidence for object detection (lower = more detections, possibly more false positives)
iou = 0.3  # IoU threshold for NMS (higher = less overlap allowed)
max_det = 20  # Maximum objects per image (increase for crowded scenes)

tracker = "./conf/trackers/bytetrack.yaml"  # Tracker config: 'bytetrack.yaml', 'botsort.yaml', etc.
track_args = {
    "persist": True,  # Keep frames history as a stream for continuous tracking
    "verbose": False,  # Print debug info from tracker
}

window_name = "Locatienet YOLO Interactive Tracking"  # Output window name

# LOGGER.info("🚀 Initializing model...")
if enable_gpu:
    # LOGGER.info("Using GPU...")
    model = YOLO(model_file)
    model.to("cuda")
else:
    # LOGGER.info("Using CPU...")
    model = YOLO(model_file, task="detect")


classes = model.names  # Store model class names

# cap = cv2.VideoCapture('videos/cars2.mp4')  # Replace with video path if needed
cap = cv2.VideoCapture('rtsp://surfguru.nl:5555')  # Replace with video path if needed




cv2.namedWindow(window_name)

fps_counter, fps_timer, fps_display = 0, time.time(), 0

LINE_ORIENTATION = 'vertical'  # 'vertical' or 'horizontal'
DIRECTION = 'positive'  # 'positive' or 'negative'
DIRECTION_POSITIVE = 'negative'  # 'positive' or 'negative'
LINE_POS = 500

counters = {
   'car' : [
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION),
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION_POSITIVE),
   ],
   'truck' : [
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION),
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION_POSITIVE),
   ],
   'motorbike' : [
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION),
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION_POSITIVE),
   ],

   'bicycle' : [
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION),
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION_POSITIVE),
   ],
    'person' : [
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION),
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION_POSITIVE),
   ],
    'dog' : [
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION),
      LineCrossCounter(line_orientation=LINE_ORIENTATION, line_pos=LINE_POS, direction=DIRECTION_POSITIVE),
   ],



}

while cap.isOpened():
    success, im = cap.read()
    if not success:
        break
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    im = cv2.resize(im, (1000, 500))

    results = model.track(im, conf=conf, iou=iou, max_det=max_det, tracker=tracker, **track_args)
    annotator = Annotator(im)
    detections = results[0].boxes.data if results[0].boxes is not None else []
    detected_objects = []
    has_counted = []
    bbox = {}
    # cv2.imwrite(f"images/{timestamp}.jpg", im.copy())
    for track in detections:
        track = track.tolist()
        if len(track) < 6:
            continue
        x1, y1, x2, y2 = map(int, track[:4])
        class_id = int(track[6]) if len(track) >= 7 else int(track[5])
        track_id = int(track[4]) if len(track) == 7 else -1
        color = colors(track_id, True)
        txt_color = annotator.get_txt_color(color)
        label = f"{classes[class_id]} ID {track_id}" + (f" ({float(track[5]):.2f})" if show_conf else "")
        class_name = model.names[class_id]
        if class_name not in counters.keys():
            continue
        for counter in counters[class_name]:
            bbox = (x1, y1, x2, y2)
            if counter.count(bbox, track_id):
                has_counted.append(class_name)

        # if save_img and len(has_counted) > 0:
        #     for class_name in has_counted:
        #         # save image with detected object
        #         cv2.imwrite(f"images/{class_name}_{timestamp}.jpg", im.copy())
        #         # write yolo labels
        #         with open(f"labels/{class_name}_{timestamp}.txt", "w") as f:
        #             img_h, img_w = im.shape[:2]
        #             x_center = ((x1 + x2) / 2) / img_w
        #             y_center = ((y1 + y2) / 2) / img_h
        #             width = (x2 - x1) / img_w
        #             height = (y2 - y1) / img_h
        #             f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


        # Draw dashed box for other objects
        for i in range(x1, x2, 10):
            cv2.line(im, (i, y1), (i + 5, y1), color, 3)
            cv2.line(im, (i, y2), (i + 5, y2), color, 3)
        for i in range(y1, y2, 10):
            cv2.line(im, (x1, i), (x1, i + 5), color, 3)
            cv2.line(im, (x2, i), (x2, i + 5), color, 3)
        # Draw label text with background
        (tw, th), bl = cv2.getTextSize(label, 0, 0.7, 2)
        cv2.rectangle(im, (x1 + 5 - 5, y1 + 20 - th - 5), (x1 + 5 + tw + 5, y1 + 20 + bl), color, -1)
        cv2.putText(im, label, (x1 + 5, y1 + 20), 0, 0.7, txt_color, 1, cv2.LINE_AA)

    if show_fps:
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        # Draw FPS text with background
        fps_text = f"FPS: {fps_display}"
        cv2.putText(im, fps_text, (10, 25), 0, 0.7, (255, 255, 255), 1)
        (tw, th), bl = cv2.getTextSize(fps_text, 0, 0.7, 2)
        cv2.rectangle(im, (10 - 5, 25 - th - 5), (10 + tw + 5, 25 + bl), (255, 255, 255), -1)
        cv2.putText(im, fps_text, (10, 25), 0, 0.7, (104, 31, 17), 1, cv2.LINE_AA)

    # Draw counting line
    if LINE_ORIENTATION == 'vertical':
        cv2.line(im,(LINE_POS, 1),(LINE_POS, 500),(0,255,0),1)
    else:
        cv2.line(im,(1, LINE_POS),(1000, LINE_POS),(0,255,0),1)

    dy = 0
    for class_name in counters.keys():
       c = 0
       for counter in counters[class_name]:
          c = c + counter.get_count()

       cvzone.putTextRect(im, f'{class_name}:{c}', (20, 75 + dy), 2, 2)
       dy = dy + 35

    if save_img and len(has_counted) > 0:
        for class_name in has_counted:
            # save image with detected object
            cv2.imwrite(f"images/{class_name}_{timestamp}.jpg", im.copy())


    cv2.imshow(window_name, im)

    # Terminal logging
    # LOGGER.info(f"🟡 DETECTED {len(detections)} OBJECT(S): {' | '.join(detected_objects)}")

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
