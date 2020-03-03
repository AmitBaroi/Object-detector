def load_yolo(directory, verbose=0):
    """
    Takes an input string specifying the directory of the YOLO weights, config and class label files and returns a
    tuple containing a list of class classes and the model.
    """
    # Class classes
    class_path = f"{directory}/coco.names"
    if verbose > 0:
        print(f"Loading class labels from '{class_path}'...")
    # Loading class labels
    classes = open(class_path).read().strip().split("\n")
    if verbose > 0:
        print(f"Contains {len(classes)} different classes.")

    from cv2.dnn import readNetFromDarknet
    # Path to model configuration and weights
    config_path = f"{directory}/yolov2.cfg"
    weight_path = f"{directory}/yolov2.weights"

    if verbose > 0:
        print(f"Loading YOLO model (config: '{config_path}', weights: '{weight_path}')...")
    # Loading model
    net = readNetFromDarknet(cfgFile=config_path, darknetModel=weight_path)

    return classes, net
