import cv2
import numpy as np
import time
import torch
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')  # or yolov5n - yolov5x6, custom

video = cv2.VideoCapture('images/Jokic.mp4')

prev_frame_time = 0
new_frame_time = 0

while (True):
    ret, frame = video.read()

    if ret == True:
        results = model(frame)
        gray = frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
    
        for i, batch_i in enumerate(results.xyxy):
            batch_i_cpu = batch_i.cpu().numpy()
            for bbox_i in batch_i_cpu:
                x1, y1, x2, y2, confidence, class_idx  = bbox_i
                cx = x1 + (x2 - x1) / 2
                cy = y1 + (y2 - y1) / 2
                output = str(confidence)
                cv2.rectangle(frame, (int(x1), int(y1)),  (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, output, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                cv2.putText(gray, fps, (7, 70), font, 3, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('Jokic 3pts',frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()