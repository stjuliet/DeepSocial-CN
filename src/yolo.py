import numpy as np
import cv2 as cv


def read_class(file):
    """ 读入目标检测类型文件，返回类别数组 """
    with open(file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes


class YoloDetect(object):
    """ 目标检测类 """
    def __init__(self, class_path, modelConfiguration, modelWeights):
        self.classes = read_class(class_path)
        self.net_input_width = 608
        self.net_input_height = 608
        self.confThreshold = 0.5
        self.nmsThreshold = 0.45
        self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        # 编译OpenCV的DNN模块使其支持Nvidia GPU请参考：
        # https://blog.csdn.net/stjuliet/article/details/107812875
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        # 无GPU
        # self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        # self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        cv.ocl.setUseOpenCL(False)

    def Init(self, frame, frame_count):
        self.frame = frame
        self.framecount = frame_count

    def getOutputsNames(self, net):
        """ 获得输出层名 """
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def drawPred(self, frame, classes, classId, conf, left, top, right, bottom):
        """ 绘制bbox """
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
        label = '%.2f' % conf
        # Get the label for the class name and its confidence
        if classes:
            assert (classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                         (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    def postprocess(self, frame, classes, outs):
        """
        后处理
        Remove the bounding boxes with low confidence using non-maximum suppression
        :param frame:
        :param classes:
        :return: outs:[507*85 =(13*13*3)*(5+80),
                 2028*85=(26*26*3)*(5+80),
                 8112*85=(52*52*3)*(5+80)]
        outs中每一行是一个预测值：[x,y,w,h,confs,class_probs_0,class_probs_1,..,class_probs_78,class_probs_79]
        :return:
        """
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []    # 保存对应二维框
        list_type = []  # 保存类型
        nms_dets = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = float(detection[0] * frameWidth)
                    center_y = float(detection[1] * frameHeight)
                    width = float(detection[2] * frameWidth)
                    height = float(detection[3] * frameHeight)
                    left = float(center_x - width / 2)
                    top = float(center_y - height / 2)

                    classIds.append(classId)
                    list_type.append(classes[classId])
                    confidences.append(float(confidence))
                    boxes.append([center_x, center_y, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            center_x = box[0]
            center_y = box[1]
            width = box[2]
            height = box[3]
            single_result = (str(list_type[i]), float(confidences[i]), (center_x, center_y, width, height))
            nms_dets.append(single_result)
            # self.drawPred(frame, classes, classIds[i], confidences[i], int(left), int(top), int(left + width), int(top + height))
        return sorted(nms_dets, key=lambda x: x[1])

    def cv_dnn_forward(self, frame):
        """
        前向传播
        :param frame:
        :return: outs:[507*85 =13*13*3*(5+80),
                       2028*85=26*26*3*(5+80),
                       8112*85=52*52*3*(5+80)]
        """
        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.net_input_width, self.net_input_height), [0, 0, 0], 1,
                                    crop=False)
        # Sets the input to the network
        self.net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutputsNames(self.net))
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        # runtime, _ = self.net.getPerfProfile()
        return outs

    def yolov3_predict(self, frame):
        """ 预测总函数 """
        outs = self.cv_dnn_forward(frame)
        # Remove the bounding boxes with low confidence
        detections = self.postprocess(frame, self.classes, outs)
        return detections
