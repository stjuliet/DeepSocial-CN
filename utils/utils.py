import numpy as np
from src.deepsocial import *


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def extract_humans(detections):
    detetcted = []
    if len(detections) > 0:  # At least 1 detection in the image and check detection presence in a frame
        idList = []
        id = 0
        for label, confidence, bbox in detections:
            if label == 'person':
                xmin, ymin, xmax, ymax = bbox2points(bbox)
                id += 1
                if id not in idList:
                    idList.append(id)
                detetcted.append([int(xmin), int(ymin), int(xmax), int(ymax), idList[-1]])
    return np.array(detetcted)


def centroid(detections, image, calibration, _centroid_dict, CorrectionShift, HumanHeightLimit):
    e = birds_eye(image.copy(), calibration)
    centroid_dict = dict()
    now_present = list()
    if len(detections) > 0:
        for d in detections:
            p = int(d[4])
            now_present.append(p)
            xmin, ymin, xmax, ymax = d[0], d[1], d[2], d[3]
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + w / 2
            y = ymax - h / 2
            if h < HumanHeightLimit:
                overley = e.image
                bird_x, bird_y = e.projection_on_bird((x, ymax))
                if CorrectionShift:
                    if checkupArea(overley, 1, 0.25, (x, ymin)):
                        continue
                e.setImage(overley)
                center_bird_x, center_bird_y = e.projection_on_bird((x, ymin))
                centroid_dict[p] = (
                    int(bird_x), int(bird_y),
                    int(x), int(ymax),
                    int(xmin), int(ymin), int(xmax), int(ymax),
                    int(center_bird_x), int(center_bird_y))

                _centroid_dict[p] = centroid_dict[p]
    return _centroid_dict, centroid_dict, e.image


def ColorGenerator(seed=1, size=10):
    np.random.seed = seed
    color = dict()
    for i in range(size):
        h = int(np.random.uniform() * 255)
        color[i] = h
    return color


def VisualiseResult(_Map, e):
    Map = np.uint8(_Map)
    histMap = e.convrt2Image(Map)
    visualBird = cv2.applyColorMap(np.uint8(_Map), cv2.COLORMAP_JET)
    visualMap = e.convrt2Image(visualBird)
    visualShow = cv2.addWeighted(e.original, 0.7, visualMap, 1 - 0.7, 0)
    return visualShow, visualBird, histMap