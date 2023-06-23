import copy
import os
import cv2 as cv
from tqdm import tqdm
from PIL import Image
import numpy as np
from base64 import b64decode, b64encode

from src.yolo import YoloDetect
from src.sort import *
from utils.utils import *

from fine_grand.fine_grand_det import CenterNet

# 目标检测配置文件(类别、模型配置、模型权重)
classesFile = "model_data/coco.names"
modelConfiguration = "model_data/yolov4.cfg"
modelWeights = "model_data/DeepSocial.weights"


if __name__ == '__main__':
    # 建立检测器和跟踪器
    yolo = YoloDetect(classesFile, modelConfiguration, modelWeights)
    mot_tracker = Sort(max_age=25, min_hits=4, iou_threshold=0.3)

    # 配置视频文件
    Input = "videos/OxfordTownCentreDataset.avi"
    calibration = [[180, 162], [618, 0], [552, 540], [682, 464]]

    # 保存演示结果文件夹路径
    save_result_dir = "results/"
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    # 配置deep social参数
    # 视频帧起始和结束
    StartFrom = 0
    EndAt = 50  # -1 表示到视频结束

    # 是否输出结果 (0表示否，1表示是）
    CouplesDetection = 1  # Enable Couple Detection
    DTC = 1  # Detection, Tracking and Couples
    SocialDistance = 1
    CrowdMap = 1
    # MoveMap             = 0
    # ViolationMap        = 0
    # RiskMap             = 0
    FineGrand = 1  # 细粒度识别+人脸识别
    if FineGrand:
        fine_grand_net = CenterNet()

    # 距离和半径设置（像素）
    ViolationDistForIndivisuals = 28
    ViolationDistForCouples = 31
    CircleradiusForIndivsual = 14
    CircleradiusForCouples = 17

    MembershipDistForCouples = (16, 10)  # (Forward, Behind) per Pixel
    MembershipTimeForCouples = 35  # Time for considering as a couple (per Frame)

    CorrectionShift = 1  # 是否忽略图像边缘的人（0表示否，1表示是）
    HumanHeightLimit = 200  # 忽略超过此高度的人
    Transparency = 0.7  # 透明度

    # 输出结果文件保存路径
    Path_For_DTC = os.path.join(save_result_dir, "DeepSOCIAL_DTC.avi")
    Path_For_SocialDistance = os.path.join(save_result_dir, "DeepSOCIAL_Social_Distancing.avi")
    Path_For_CrowdMap = os.path.join(save_result_dir, "DeepSOCIAL_Crowd_Map.avi")

    # 读入视频
    cap = cv.VideoCapture(Input)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if EndAt == -1:
        EndAt = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # 获得视频总帧数
    # 保存视频的分辨率下降倍数
    ReductionFactor = 2
    # 保存视频的分辨率
    height, width = frame_height // ReductionFactor, frame_width // ReductionFactor

    # 保存为视频格式
    if DTC:
        DTCVid = cv.VideoWriter(Path_For_DTC, cv.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
    if SocialDistance:
        SDimageVid = cv.VideoWriter(Path_For_SocialDistance, cv.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
    if CrowdMap:
        CrowdVid = cv.VideoWriter(Path_For_CrowdMap, cv.VideoWriter_fourcc(*"XVID"), 30.0, (width, height))

    colorPool = ColorGenerator(size=3000)
    _centroid_dict = dict()
    _numberOFpeople = list()
    _greenZone = list()
    _redZone = list()
    _yellowZone = list()
    _final_redZone = list()
    _relation = dict()
    _couples = dict()
    _trackMap = np.zeros((height, width, 3), dtype=np.uint8)
    _crowdMap = np.zeros((height, width), dtype=np.int)
    _allPeople = 0
    _counter = 1
    frame_count = 0

    # 检测
    video_length = EndAt-StartFrom
    with tqdm(total=video_length, desc=f"Process", mininterval=0.3) as pbar:
        for _ in range(video_length):
            ret, frame_read = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count <= StartFrom:
                continue
            if frame_count > EndAt:
                break
            frame_resized = cv.resize(frame_read, (width, height), interpolation=cv.INTER_LINEAR)
            image = frame_resized

            e = birds_eye(image, calibration)

            yolo.Init(image, frame_count)
            detections = yolo.yolov3_predict(image)
            # 中心宽高形式 -> 左上右下形式！
            humans = extract_humans(detections)

            # 使得检测出的人的边框不超出图像范围
            humans[:, 0:4:2] = np.clip(humans[:, 0:4:2], 0, width - 1)
            humans[:, 1:4:2] = np.clip(humans[:, 1:4:2], 0, height - 1)

            if FineGrand:
                for human in humans:
                    left, top, right, bottom = human[:-1]
                    if left < right - 1 and top < bottom - 1:  # 保证能裁剪出人像
                        crop_image = image[top:bottom, left:right]
                        # opencv -> pil
                        crop_pil_image = Image.fromarray(np.uint8(cv.cvtColor(crop_image, cv.COLOR_BGR2RGB)))
                        results = fine_grand_net.detect_image(crop_pil_image)
                        _, attrib_dict, boxes, labels, confs = results
                        if boxes is not None:
                            box = boxes[0]
                            bt, bl, bb, br = box
                            # 变换至原始图像坐标
                            cv.rectangle(image, (int(left + bl), int(top + bt)), (int(left + br), int(top + bb)), (0, 0, 255), 2)
                            # 取出细粒度属性字典，按每一个属性往下排列显示
                            for i, (k, v) in enumerate(attrib_dict.items()):
                                attrib_str = (str(k) + ": " + str(v))
                                str_size = cv.getTextSize(attrib_str, cv.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]  # w, h
                                cv.putText(image, attrib_str, (int(left + bl), int(bottom + bt + i * str_size[1] + 3)),
                                           cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

            track_bbs_ids = mot_tracker.update(humans) if len(humans) != 0 else humans

            _centroid_dict, centroid_dict, partImage = centroid(track_bbs_ids, image, calibration, _centroid_dict,
                                                                CorrectionShift, HumanHeightLimit)
            redZone, greenZone = find_zone(centroid_dict, _greenZone, _redZone, criteria=ViolationDistForIndivisuals)

            if CouplesDetection:
                _relation, relation = find_relation(e, centroid_dict, MembershipDistForCouples, redZone, _couples, _relation)
                _couples, couples, coupleZone = find_couples(image, _centroid_dict, relation, MembershipTimeForCouples,
                                                             _couples)
                yellowZone, final_redZone, redGroups = find_redGroups(image, centroid_dict, calibration,
                                                                      ViolationDistForCouples, redZone, coupleZone, couples,
                                                                      _yellowZone, _final_redZone)
            else:
                couples = []
                coupleZone = []
                yellowZone = []
                redGroups = redZone
                final_redZone = redZone

            if DTC:
                DTC_image = image.copy()
                _trackMap = Apply_trackmap(centroid_dict, _trackMap, colorPool, 3)
                DTC_image = cv.add(e.convrt2Image(_trackMap), image)
                DTCShow = DTC_image
                for id, box in centroid_dict.items():
                    center_bird = box[0], box[1]
                    if not id in coupleZone:
                        cv.rectangle(DTCShow, (box[4], box[5]), (box[6], box[7]), (0, 255, 0), 2)
                        cv.rectangle(DTCShow, (box[4], box[5] - 13), (box[4] + len(str(id)) * 10, box[5]), (0, 200, 255), -1)
                        cv.putText(DTCShow, str(id), (box[4] + 2, box[5] - 2), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1, cv.LINE_AA)
                for coupled in couples:
                    p1, p2 = coupled
                    couplesID = couples[coupled]['id']
                    couplesBox = couples[coupled]['box']
                    cv.rectangle(DTCShow, couplesBox[2:4], couplesBox[4:], (0, 150, 255), 4)
                    loc = couplesBox[0], couplesBox[3]
                    offset = len(str(couplesID) * 5)
                    captionBox = (loc[0] - offset, loc[1] - 13), (loc[0] + offset, loc[1])
                    cv.rectangle(DTCShow, captionBox[0], captionBox[1], (0, 200, 255), -1)
                    wc = captionBox[1][0] - captionBox[0][0]
                    hc = captionBox[1][1] - captionBox[0][1]
                    cx = captionBox[0][0] + wc // 2
                    cy = captionBox[0][1] + hc // 2
                    textLoc = (cx - offset, cy + 4)
                    cv.putText(DTCShow, str(couplesID), (textLoc), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1, cv.LINE_AA)
                DTCVid.write(DTCShow)

            if SocialDistance:
                SDimage, birdSDimage = Apply_ellipticBound(centroid_dict, image, calibration, redZone, greenZone, yellowZone,
                                                           final_redZone, coupleZone, couples, CircleradiusForIndivsual,
                                                           CircleradiusForCouples)
                SDimageVid.write(SDimage)

            if CrowdMap:
                _crowdMap, crowdMap = Apply_crowdMap(centroid_dict, image, _crowdMap)
                crowd = (crowdMap - crowdMap.min()) / (crowdMap.max() - crowdMap.min()) * 255
                crowd_visualShow, crowd_visualBird, crowd_histMap = VisualiseResult(crowd, e)
                CrowdVid.write(crowd_visualShow)

            cv.waitKey(3)

            pbar.update(1)
        print('::: Analysis Completed')
        cap.release()
        if DTC:
            DTCVid.release()
            print("::: Video Write Completed : ", Path_For_DTC)
        if SocialDistance:
            SDimageVid.release()
            print("::: Video Write Completed : ", Path_For_SocialDistance)
        if CrowdMap:
            CrowdVid.release()
            print("::: Video Write Completed : ", Path_For_CrowdMap)
