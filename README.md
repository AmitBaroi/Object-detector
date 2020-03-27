# Object-detector
This is a simple program that detects objects in static images or live webcam video feed.
It then highlights the bouding boxes and names of those obejcts in unique colors.
For object detection the [YOLOv2 model](https://pjreddie.com/darknet/yolov2/), trained on the [COCO dataset](http://cocodataset.org/) was used.
The COCO dataset contains 80 different object classes. The names of the object classes are available [here](https://github.com/pjreddie/darknet/blob/master/data/coco.names).

![object-detection](detection.png)

#### How to run the program:
1. Download this repository by clicking the green "Clone or download" button on the top left. 
2. Wait for the **Object-detector-master** folder to finish downloading (extract the compressed file, if you downloaded as zip).
3. Download the YOLOv2 model weights from [this link](https://pjreddie.com/media/files/yolov2.weights) and put the file in the **Object-detector-master/yolo-coco/** directory.
4. Open the terminal inside the **Object-detector-master** folder and enter the following command: `python yolo.py`.
5. Select appropriate options for images or webcam feed.

ENJOY!

Alternatively, you could also run `yolo_img.py` or `yolo_cam.py` if you want.
