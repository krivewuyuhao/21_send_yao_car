import albumentations as A
import cv2
import os

# 设置需要使用的数据增强方式
transform = A.Compose([
    A.RandomScale(scale_limit=(0.5, 1.5), p=0.5), # 随机缩放
    A.Rotate(limit=[-45, 45], p=0.5),   # 随机旋转，旋转角度在-70到70度之间，有50%的概率被应用
    A.RandomBrightnessContrast(p=0.3),   # 随机调整亮度和对比度，有30%的概率被应用
    A.Resize(128,128,),
])

# 遍历每一个数字文件夹
for i in range(1, 9):
    # 图片计数
    counts = 0
    img_list = os.listdir(f"data_set/{i}")
    for temp in img_list:
        image = cv2.imread(f"data_set/{i}/{temp}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transformed = transform(image=image)['image']
        # 保存数据增强后的图片
        cv2.imwrite(f"data_set/{i}/{i}__{counts}__E.jpg", transformed)
        counts += 1
