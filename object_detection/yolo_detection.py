import torch

# Load a pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can also try 'yolov5m', 'yolov5l', 'yolov5x' for larger models

# Load an image
img = 'https://ultralytics.com/images/zidane.jpg'  # Or use a local image path

# Perform object detection
results = model(img)

# Print the results
results.print()

# Save the results with bounding boxes
results.save(save_dir='runs/detect')