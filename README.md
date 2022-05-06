# DeepSocial-CN
根据原作者提供的代码添加中文注释，并且增加OpenCV-DNN模块进行YOLOv4目标检测。
## 使用方法
- Colab Notebook版本：
  
  https://github.com/stjuliet/DeepSocial-CN/blob/master/deepsocial_juliet.ipynb

- OpenCV本地版本：

1. 克隆Repo:
```
git clone https://github.com/stjuliet/DeepSocial-CN
```

2. 安装依赖项:
```
cd ~/DeepSocial-CN
pip install -r requirements.txt
```

3. 编译Nvidia GPU版本的OpenCV-DNN模块，进行YOLOv4目标检测:
```
参考：https://blog.csdn.net/stjuliet/article/details/107812875
```
注：如果不需要使用GPU加速，可将src/yolo.py文件中的24、25两行改成：
```
self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
```

4. 修改参数，运行```yolov4_deepsocial.py```

## 论文和代码下载
DeepSOCIAL: Social Distancing Monitoring and Infection Risk Assessment in COVID-19 Pandemic

Open access paper: https://www.mdpi.com/2076-3417/10/21/7514 & https://doi.org/10.3390/app10217514

Code: https://github.com/DrMahdiRezaei/DeepSOCIAL

## 测试视频和模型文件下载
https://github.com/stjuliet/DeepSocial-CN/releases/tag/video_code
