# 定义预处理工具
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def preprocess_image(image):
    # mean = [0.40789655, 0.44719303, 0.47026116]
    # std = [0.2886383, 0.27408165, 0.27809834]
    mean = [0.484, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return ((np.float32(image) / 255.) - mean) / std


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def GetAttriPool(train_lines):
    '''
    [
            '朝向：正', '朝向：背', '朝向：左', '朝向：右',
        '性别：男', '性别：女', '性别：不确定',
        '年龄：0-18', '年龄：18-55','年龄：>55', '年龄：不确定',
        '上身颜色：其他', '上身颜色：黑', '上身颜色：白', '上身颜色：灰', '上身颜色：红',
                '上身颜色：黄', '上身颜色：绿', '上身颜色：蓝', '上身颜色：紫',
                '上身颜色：棕', '上身颜色：粉', '上身颜色：橙',
        '下身颜色：其他', '下身颜色：黑', '下身颜色：白', '下身颜色：灰', '下身颜色：红',
                '下身颜色：黄', '下身颜色：绿', '下身颜色：蓝', '下身颜色：紫',
                '下身颜色：棕', '下身颜色：粉', '下身颜色：橙'
                ]
    '''
    n = len(train_lines)
    pool = {}
    for i in range(n):
        line = train_lines[i]
        items = line.strip().split(' ')
        attris = items[-1].split(',')


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def centernet_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes


def resize_centernet_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    # input_shape[0] = h, input_shape[1] = w; now
    # image_shape[0] = h, image_shape[1] = w; ori
    scale_w = input_shape[1] / image_shape[1]
    scale_h = input_shape[0] / image_shape[0]
    return [top/scale_h, left/scale_w, bottom/scale_h, right/scale_w]


def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def decode_bbox(pred_hms, pred_whs, pred_offsets, threshold, topk=100):
    #-------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 80, 128, 128, 
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    #-------------------------------------------------------------------------#
    pred_hms = pool_nms(pred_hms)
    
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    for batch in range(b):
        #-------------------------------------------------------------------------#
        #   heat_map        128*128, num_classes    热力图
        #   pred_wh         128*128, 2              特征点的预测宽高
        #   pred_offset     128*128, 2              特征点的xy轴偏移情况
        #-------------------------------------------------------------------------#
        heat_map    = pred_hms[batch].permute(1,2,0).reshape([-1,c])
        pred_wh     = pred_whs[batch].permute(1,2,0).reshape([-1,2])
        pred_offset = pred_offsets[batch].permute(1,2,0).reshape([-1,2])

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        #print(xv)
        #-------------------------------------------------------------------------#
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        #-------------------------------------------------------------------------#
        xv, yv = xv.flatten().float(), yv.flatten().float()

        #-------------------------------------------------------------------------#
        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        #-------------------------------------------------------------------------#
        class_conf, class_pred = torch.max(heat_map, dim=-1)
        mask = class_conf > threshold
        #-----------------------------------------#
        #   取出得分筛选后对应的结果
        #-----------------------------------------#
        pred_wh_mask = pred_wh[mask]
        pred_offset_mask = pred_offset[mask]
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue

        # paddle写法，无法直接索引tensor，因此较为复杂
        # mask_indexes = layers.where(mask).numpy() #.squeeze()
        # mask_indexes = list(mask_indexes)
        # if len(mask_indexes) > 50:
        #     mask_indexes = mask_indexes[:50]
        #
        # pred_wh_mask = []
        # for i in mask_indexes:
        #     pred_wh_mask.append(pred_wh[int(i):int(i+1), :])
        # if len(pred_wh_mask)!=0:
        #     pred_wh_mask = paddle.concat(pred_wh_mask)
        # else:
        #     detects.append([])
        #     continue
        #
        # pred_offset_mask = []
        # for i in mask_indexes:
        #     pred_offset_mask.append(pred_offset[int(i):int(i+1), :])
        # pred_offset_mask = paddle.concat(pred_offset_mask)
        # if len(pred_wh_mask)==0:
        #     detects.append([])
        #     continue

        #----------------------------------------#
        #   计算调整后预测框的中心
        #----------------------------------------#
        # xv_mask = []
        # for i in mask_indexes:
        #     xv_mask.append(xv[int(i)])
        # xv_mask = paddle.concat(xv_mask)
        #
        # yv_mask = []
        # for i in mask_indexes:
        #     yv_mask.append(yv[int(i)])
        # yv_mask = paddle.concat(yv_mask)

        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
        #----------------------------------------#
        #   计算预测框的宽高
        #----------------------------------------#
        half_w, half_h = pred_wh_mask[:, 0:1] / 2, pred_wh_mask[:, 1:2] / 2
        #----------------------------------------#
        #   获得预测框的左上角和右下角
        #----------------------------------------#
        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h
        detect = torch.cat(
            [bboxes, torch.unsqueeze(class_conf[mask], -1), torch.unsqueeze(class_pred[mask], -1).float()], dim=-1)
        detects.append(detect)
        # bboxes = paddle.concat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], axis=1)
        # bboxes[:, 0] /= output_w
        # bboxes[:, 1] /= output_h
        # bboxes[:, 2] /= output_w
        # bboxes[:, 3] /= output_h
        #
        # class_conf_mask = []
        # for i in mask_indexes:
        #     class_conf_mask.append(class_conf[int(i)])
        # class_conf_mask = paddle.concat(class_conf_mask)
        #
        # class_pred_mask = []
        # for i in mask_indexes:
        #     class_pred_mask.append(class_pred[int(i)])
        # class_pred_mask = paddle.concat(class_pred_mask)
        #
        # detect = paddle.concat(
        #     [bboxes, paddle.unsqueeze(class_conf_mask,-1),
        #     paddle.unsqueeze(class_pred_mask,-1).astype('float32')], axis=-1
        # )
        #
        # arg_sort = paddle.argsort(detect[:,-2], descending=True)
        # # print(detect.shape) # [50, 6]
        # detect_mask = []
        # for i in arg_sort:
        #     detect_mask.append(detect[int(i):int(i)+1, :])
        # detect_mask = paddle.concat(detect_mask)
        #
        # detect = detect_mask
        # detects.append(detect.numpy()[:topk])
        
    return detects


def nms(results, nms):
    outputs = []
    # 对每一个图片进行处理
    for i in range(len(results)):
        #------------------------------------------------#
        #   具体过程可参考
        #   https://www.bilibili.com/video/BV1Lz411B7nQ
        #------------------------------------------------#
        detections = results[i]
        unique_class = np.unique(detections[:,-1])

        best_box = []
        if len(unique_class) == 0:
            results.append(best_box)
            continue
        #-------------------------------------------------------------------#
        #   对种类进行循环，
        #   非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框，
        #   对种类进行循环可以帮助我们对每一个类分别进行非极大抑制。
        #-------------------------------------------------------------------#
        for c in unique_class:
            cls_mask = detections[:,-1] == c

            detection = detections[cls_mask]
            scores = detection[:,4]
            # 根据得分对该种类进行从大到小排序。
            arg_sort = np.argsort(scores)[::-1]
            detection = detection[arg_sort]
            while np.shape(detection)[0]>0:
                # 每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
                best_box.append(detection[0])
                if len(detection) == 1:
                    break
                ious = iou(best_box[-1],detection[1:])
                detection = detection[1:][ious<nms]
        outputs.append(best_box)
        # print(best_box[0].shape)
    return outputs


def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    
    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
    
    iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
    return iou


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)
