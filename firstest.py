import cv2
import torch
import os

# 1. Path to your cloned yolov5 folder and your model
yolo_repo_path = 'yolov5'  # The folder name created by git clone
model_path = 'C:\\PPE DETECTION\\best .pt' # Ensure no space in filename

# 2. Load the model LOCALLY (source='local')
# This avoids the "trusted repository" prompt and uses your cloned files
model = torch.hub.load(yolo_repo_path, 'custom', path=model_path, source='local')

# 3. Initialize Webcam
cap = cv2.VideoCapture(0)

print("Starting Local YOLOv5... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:#press q to stop
        break

    # 4. Inference
    results = model(frame)

    # 5. Render results on the frame
    # results.render() returns a list of images, we take the first one [0]
    results.render() 
    annotated_frame = results.ims[0] 

    # 6. Show the window
    cv2.imshow('Local PPE Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()