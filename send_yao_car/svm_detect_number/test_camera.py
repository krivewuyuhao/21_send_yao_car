import cv2
import imutils
from pynput.keyboard import Key, Listener
import joblib


# 加载SVM和PCA模型
loaded_svm = joblib.load('best_svm_model.joblib')
loaded_pca = joblib.load('best_pca_model.joblib')

# 显示字体配置
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
color = (0, 255, 0)

cap = cv2.VideoCapture(0)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
while True:
    flag, frame = cap.read()
    if frame is None:
        continue
    if flag is True:
        frame_copy = frame.copy()
        frame_copy_2 = frame.copy()
        # 自动阈值处理
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_ths = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
        # 闭操作
        img_closed = cv2.morphologyEx(frame_ths, cv2.MORPH_CLOSE, rectKernel)
        # canny边缘检测
        temp = imutils.auto_canny(img_closed)
        # 轮廓查找
        contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 计算轮廓REE
        for index, c in enumerate(contours):
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if area > 20000:
                img_out = cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                res_img = frame_copy_2[y:y+h, x:x+w]
                # 图像预处理
                img = cv2.resize(res_img, (128, 128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                temp = img.reshape(1, -1)
                # PCA降维
                temp = loaded_pca.transform(temp)
                y_new_pred = loaded_svm.predict(temp)
                print(y_new_pred)
                cv2.putText(frame_copy, f"Pred: {y_new_pred}", (int(x), int(y)),
                            font, font_scale, color, font_thickness)

        cv2.imshow("frame", frame_copy)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

