import numpy as np
import time
import cv2
import os
from utils import *

np.random.seed(42)

CONF = 0.5  # Confidence
THRESH = 0.3  # Threshold
MODEL_DIR = "yolo-coco"
IMG_DIR = "images"
MODE = input("""Select object detection mode:
- images
- webcam
>>> """).lower()

# Loading class labels and YOLO model
labels, model = load_yolo(MODEL_DIR)
# Output layer names needed from YOLO
ln = model.getLayerNames()
out_layers = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]
# Generating some colors for each class
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

if MODE in ["images", "image"]:
    file_names = os.listdir(IMG_DIR)
    file_paths = [os.path.join(IMG_DIR, file) for file in os.listdir(IMG_DIR)]
    # Iterate over all files in image directory
    for image_path in file_paths:
        print("-" * 40)
        print(f"Processing: '{image_path}'")
        print("-" * 40)
        # Reading in image as numpy array
        image = cv2.imread(image_path)
        # Image height and width
        (height, width) = image.shape[:2]

        # # # OBJECT DETECTION # # #
        layer_outputs = detect_objects(image, model, out_layers)
        boxes, confidences, class_ids = predict_bboxes(layer_outputs, width, height)
        # Applying non-max suppression to suppress weak overlapping bounding boxes
        ids = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=CONF, nms_threshold=THRESH)
        print(f"Detections kept after non-max suppression: {len(ids)}")
        # Drawing bounding boxes
        image = draw_boxes(image, ids, boxes, confidences, colors, class_ids, labels)

        # Display output image
        cv2.imshow(f"Image: {os.path.split(image_path)[1]}", image)
        cv2.waitKey(0)
elif MODE in ["webcam", "web-cam"]:
    cap = cv2.VideoCapture(0)
    while True:
        # Reading in each frame as an image
        _, image = cap.read()
        # Image height and width
        (height, width) = image.shape[:2]

        # # # OBJECT DETECTION # # #
        layer_outputs = detect_objects(image, model, out_layers)
        boxes, confidences, class_ids = predict_bboxes(layer_outputs, width, height)
        # Applying non-max suppression to suppress weak overlapping bounding boxes
        ids = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=CONF, nms_threshold=THRESH)
        print(f"Detections kept after non-max suppression: {len(ids)}")
        # Drawing bounding boxes
        image = draw_boxes(image, ids, boxes, confidences, colors, class_ids, labels)

        # Display output image
        cv2.imshow(f"Live object detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
else:
    print("INVALID INPUT!")

cv2.destroyAllWindows()
