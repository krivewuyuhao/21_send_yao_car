"""
function : Image preprocessing
"""


import cv2

cap = cv2.VideoCapture(0)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))

while True:
    flag, frame = cap.read()
    # 如果正确读帧
    if frame is None:
        print("The camera is not open")
        continue

    if flag is True:
        frame_copy = frame.copy()   # 展示面积
        frame_copy1 = frame.copy()
        frame_copy2 = frame.copy()  # 截取的矩形框
        # 转为灰度图
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 大津法处理阈值，背景反色，采用索引取出图像
        # 预留回调函数，手动调节阈值thresh
        ths_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV and cv2.THRESH_OTSU)[1]
        cv2.imshow('ths_frame', ths_frame)
        # 形态学操作
        img_closed = cv2.morphologyEx(ths_frame, cv2.MORPH_CLOSE, rectKernel)
        # 边缘检测
        img_blur = cv2.GaussianBlur(img_closed, (3, 3), 0)
        img_dst = cv2.Canny(img_blur, 100, 200)
        # 轮廓查找
        contours, hierarchy = cv2.findContours(img_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for region in contours:
            area = cv2.contourArea(region)
            x, y, w, h = cv2.boundingRect(region)
            if area > 10000:
                frame_copy = cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # 获取轮廓区域的图像
                res_img = frame_copy2[y:y + h, x:x + w]
                cv2.putText(frame_copy, f'Area: {area:.2f}', (x, y - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        # 显示
        cv2.imshow("find the contours", frame_copy)
        cv2.imshow("closed", img_closed)
        cv2.imshow('截取的矩形', frame_copy2)

        # 退出按键
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            break