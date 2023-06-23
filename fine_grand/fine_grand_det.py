import colorsys
import torch
import os
import numpy as np
from .model import CenterNet_ResNet
from .utils import preprocess_image
from .decode_box import decode_bbox, postprocess
from PIL import Image, ImageDraw


# 预测时使用
class CenterNet(object):
    _defaults = {
        "pretrained": False,
        "model_path": 'fine_grand/work/r18_stage2_dynamic/Epoch27-Total_Loss0.7426-Val_Loss0.7817',
        "classes_path": 'fine_grand/work/model_data/pede.txt',
        "n_class": 26,
        "image_size": [256, 128, 3],
        "backbone": 'resnet_18',
        "confidence"        : 0.3,
        "nms"               : True,
        "nms_threhold"      : 0.3,
        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化centernet
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        # ----------------------------------------#
        #   计算种类数量
        # ----------------------------------------#
        self.num_classes = len(self.class_names)

        # ----------------------------------------#
        #   创建centernet模型
        # ----------------------------------------#
        self.centernet = CenterNet_ResNet(num_classes=self.num_classes, config=self.__dict__)  # 动态图加载方式
        state_dict = torch.load(self.model_path + '.pth')
        self.centernet.load_state_dict(state_dict)
        # self.centernet = paddle.jit.load(self.model_path)  # 静态图模型直接加载

        self.centernet = self.centernet.eval()
        self.centernet = self.centernet.cuda()

        print('{} model, and classes loaded.'.format(self.model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        self.attribute_list = [
            'sex: female', 'age: >60', 'age: 18-60', 'age: <18',
            'front', 'side', 'back',
            'hat', 'glasses', 'HandBag', 'ShoulderBag', 'BackPack',
            'HoldObjectsInFront', 'ShortSleeve', 'LongSleeve',
            'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice',
            'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers',
            'Shorts', 'Skirt&Dress', 'boots'
        ]

        self.mapp = [1,3,3,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1]

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])  # hwc

        crop_img = image.resize((self.image_size[1], self.image_size[0]))
        # ----------------------------------------------------------------------------------#
        #   将RGB转化成BGR，这是因为原始的centernet_hourglass权值是使用BGR通道的图片训练的
        # ----------------------------------------------------------------------------------#
        photo = np.array(crop_img, dtype=np.float32)[:, :, ::-1]
        # -----------------------------------------------------------#
        #   图片预处理，归一化。
        # -----------------------------------------------------------#
        photo = np.reshape(np.transpose(preprocess_image(photo), (2, 0, 1)),
                           [1, self.image_size[2], self.image_size[0], self.image_size[1]])

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(photo)).type(torch.FloatTensor).cuda()

            outputs = self.centernet(images)
            # 解析细粒度属性
            attris = outputs[-1].detach().cpu().numpy()[0]
            attris = np.where(attris >= 0.5, 1, 0)
            attris_list = list(attris)

            sum_index = 0
            attris_dict = {}
            for index in self.mapp:
                charact = np.argmax(np.array(attris_list[sum_index:sum_index+index]))
                label = self.attribute_list[sum_index:sum_index+index]
                sum_index += index
                attris_dict[label[charact]] = charact

            # 解析box
            # -----------------------------------------------------------#
            #   利用预测结果进行解码
            # -----------------------------------------------------------#
            output_box = outputs[0]
            output_box = decode_bbox(output_box[0], output_box[1], output_box[2], self.confidence)
            results = postprocess(output_box, self.nms, image_shape, self.image_size, self.letterbox_image, self.nms_threhold)

            # --------------------------------------#
            #   如果没有检测到物体，则返回原图
            # --------------------------------------#
            if results[0] is None:
                return image, attris_dict, None, None, None

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

            # for i, c in list(enumerate(top_label)):
            #     predicted_class = self.class_names[int(c)]
            #     box = top_boxes[i]
            #     score = top_conf[i]
            #
            #     top, left, bottom, right = box
            #
            #     top = max(0, np.floor(top).astype('int32'))
            #     left = max(0, np.floor(left).astype('int32'))
            #     bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            #     right = min(image.size[0], np.floor(right).astype('int32'))
            #
            #     label = '{} {:.2f}'.format(predicted_class, score)
            #     draw = ImageDraw.Draw(image)
            #     # label_size = draw.textsize(label, font)
            #     label = label.encode('utf-8')
            #     # print(label, top, left, bottom, right)
            #
            #     # if top - label_size[1] >= 0:
            #     #     text_origin = np.array([left, top - label_size[1]])
            #     # else:
            #     #     text_origin = np.array([left, top + 1])
            #
            #     for i in range(2):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #     # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #     # draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            #     del draw

        return image, attris_dict, top_boxes, top_label, top_conf


if __name__ == "__main__":
    # 执行预测
    import os
    from tqdm import tqdm
    from PIL import Image
    import cv2 as cv
    import numpy as np

    centernet = CenterNet()
    test_file_txt = "pa100k/test_1114.txt"
    img_root = "pa100k/release_data/release_data"
    with open(test_file_txt, "r") as fread:
        test_file_path_list = fread.readlines()
        with tqdm(total=len(test_file_path_list), postfix=dict) as pbar:
            for single_test_file_path in test_file_path_list:
                image = Image.open(os.path.join(img_root, single_test_file_path.split(" ")[0]))
                r_image, attibs_dict, box, label, conf = centernet.detect_image(image)
                print(attibs_dict)
                img = cv.cvtColor(np.asarray(r_image), cv.COLOR_RGB2BGR)  # 转换代码
                cv.imshow('img', img)  # opencv显示
                cv.waitKey()
