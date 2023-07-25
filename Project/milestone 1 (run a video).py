import cv2
import torch
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')  # or yolov5n - yolov5x6, custom

video = cv2.VideoCapture('images/Jokic.mp4')

while (True):
    ret, frame = video.read()

    if ret == True:
        cv2.imshow('Jokic 3pts',frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
video.release()
cv2.destroyAllWindows()