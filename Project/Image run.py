import cv2
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')  # or yolov5n - yolov5x6, custom

img = cv2.imread("images/Screenshot 2023-07-19 190807.png",cv2.IMREAD_UNCHANGED)


results = model(img)
for i, batch_i in enumerate(results.xyxy):
    batch_i_cpu = batch_i.cpu().numpy()
    for bbox_i in batch_i_cpu:
        x1, y1, x2, y2, confidence, class_idx  = bbox_i
        cx = x1 + (x2 - x1) / 2
        cy = y1 + (y2 - y1) / 2
        output = str(confidence)
        cv2.rectangle(img, (int(x1), int(y1)),  (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, output, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        


cv2.imshow('temp', img)
cv2.waitKey(0)