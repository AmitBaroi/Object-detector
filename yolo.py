import numpy as np
import time
import cv2
import os
from utils import *
np.random.seed(42)

# Path to test image directory
IMG_DIR = "images"
# Loading class labels and YOLOv3 model
LABELS, MODEL = load_yolo("yolo-coco")
# Generating some colors for each class
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# List of images in image directory
image_files = [os.path.join(IMG_DIR, file) for file in os.listdir(IMG_DIR)]

# TODO: update to iterate over all files in image directory
image_path = image_files[0]
print(f"Processing '{image_path}'...")

# Reading in image as numpy array
image = cv2.imread(image_path)
# Image height (H) and width (W)
(H, W) = image.shape[:2]

# Output layer names needed from YOLOv3
ln = MODEL.getLayerNames()
ln = [ln[i[0]-1] for i in MODEL.getUnconnectedOutLayers()]

# Generate blob from input image, do forward pass with YOLO detector, give bounding boxes and probabilities
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(416, 416), swapRB=True, crop=False)
MODEL.setInput(blob)
start = time.time()
layer_outputs = MODEL.forward(ln)
end = time.time()
print(f"YOLOv3 prediction took {np.round(end-start, 6)} seconds")

# TODO: iterate over layer_outputs and draw prediction and bbox on output image

# Display output image
cv2.imshow("Image", image)
cv2.waitKey(0)

