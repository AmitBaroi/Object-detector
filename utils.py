def load_yolo(directory, verbose=0):
    """
    Takes an input string specifying the directory of the YOLOv3 weights, config and class label files and returns a
    tuple containing a list of class classes and the model.
    """
    # Class classes
    class_path = f"{directory}/coco.names"
    if verbose > 0:
        print(f"Loading class labels from '{class_path}'...")
    classes = open(class_path).read().strip().split("\n")
    if verbose > 0:
        print(f"Contains {len(classes)} different classes.")

    # Model
    from cv2.dnn import readNetFromDarknet
    config_path = f"{directory}/yolov3.cfg"
    weight_path = f"{directory}/yolov3.weights"
    if verbose > 0:
        print(f"Loading YOLOv3 model (config: '{config_path}', weights: '{weight_path}')...")
    net = readNetFromDarknet(cfgFile=config_path, darknetModel=weight_path)

    return classes, net
