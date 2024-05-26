import joblib
import cv2
import numpy as np


# 加载SVM分类模型
loaded_svm = joblib.load('best_svm_model.joblib')
# 加载PCA降维模型
loaded_pca = joblib.load('best_pca_model.joblib')

# 读取图片并进行预处理
img = cv2.imread("08.jpg")
img_test = img.copy()
cv2.imshow('img', img_test)

img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

temp = img.reshape(1, -1)
# PCA降维，保证格式与训练数据相同
test_data_pca = loaded_pca.transform(temp)
print(test_data_pca.shape)
y_new_predict = loaded_svm.predict(test_data_pca)
predictions = loaded_svm.predict(test_data_pca)

# 输出预测结果
print("Predictions for new data:", y_new_predict)
accuracy = np.mean(predictions == [8])
print("Accuracy on test data:", accuracy)

'''
在图中标出准确率
'''
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11), (-1, -1))
# 转为灰度图
img_copy = img_test.copy()
gray_frame = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
# 大津法处理阈值，背景反色，采用索引取出图像
# 预留回调函数，手动调节阈值thresh
ths_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# 形态学操作
img_closed = cv2.morphologyEx(ths_frame, cv2.MORPH_CLOSE, rectKernel)
# 边缘检测
img_blur = cv2.GaussianBlur(img_closed, (3, 3), 0)
img_temp = cv2.Canny(img_blur, 100, 200)
# 轮廓查找
contours, hierarchy = cv2.findContours(img_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 初始化最大面积和对应的索引
max_area = 0
max_index = -1
for index, region in enumerate(contours):
    area = cv2.contourArea(region)

    if area > max_area:
        max_area = area
        max_index = index

# 如果找到最大轮廓，则在该轮廓上绘制矩形和文本
if max_index != -1:
    region = contours[max_index]
    x, y, w, h = cv2.boundingRect(region)

    # 绘制矩形框
    img_out = cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 获取轮廓区域的图像
    res_img = img_copy[y:y + h, x:x + w]

    # 假设您已经有了一个准确率的数值，这里显示为accuracy
    # 请确保accuracy已经被定义，比如accuracy = 0.95
    cv2.putText(img_copy, f'Accuracy: {accuracy:.2f}', (x, y - 20),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("test", img_copy)

cv2.waitKey()

