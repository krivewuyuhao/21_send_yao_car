import array
import math

import cv2
import imutils
import joblib
import cv2
import numpy as np
import time
# import serial

def sendMessage(msg):
    ser.write(msg.encode("ascii"))

def on_trackbar(val, trackbar_name):
    if trackbar_name == 'Brightness':
        cap.set(cv2.CAP_PROP_BRIGHTNESS, val)
    elif trackbar_name == 'Contrast':
        cap.set(cv2.CAP_PROP_CONTRAST, val)
    elif trackbar_name == 'Saturation':
        cap.set(cv2.CAP_PROP_SATURATION, val)
    elif trackbar_name == 'Hue':
        cap.set(cv2.CAP_PROP_HUE, val)
    print(f"{trackbar_name} set to {val}")
    pass

def create_trackbar(name, max_range, default_value):
    cv2.createTrackbar(name, "src", default_value, max_range, lambda val: on_trackbar(val, name))

cap = cv2.VideoCapture(0)

# 设置波特率便于通信
# ser = serial.Serial(port="/dev/ttyAMA0", baudrate=115200, timeout=0)

# 显示字体配置
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

# 显示预测的颜色
show_color = (0, 255, 0)
'''
定义三种状态：
1. 开始时候的数字识别A
2. 巡线 B
3. 十字路口数字识别 C

'''

'''
加载模型
'''
# 加载SVM和PCA模型
loaded_svm = joblib.load('svm_detect_number/best_svm_model.joblib')
loaded_pca = joblib.load('svm_detect_number/best_pca_model.joblib')

'''
转化到hsv空间下处理红色的阈值,具体怎么处理反光问题？该看看现场环境
'''
color_x = color_y = color_radius = 0
color_hsv = {"red": ((0, 100, 100), (10, 255, 255))}   # 红色像素在hsv空间下的阈值范围
color = "red"
color_lower = np.array([color_hsv[color][0][0], color_hsv[color][0][1], color_hsv[color][0][2]])
color_upper = np.array([color_hsv[color][1][0], color_hsv[color][1][1], color_hsv[color][1][2]])

'''创建一个窗口，用交互式方式调节摄像头'''
cv2.namedWindow("src")
# 创建滑动条并关联回调函数
create_trackbar('Brightness',  100, 59)
create_trackbar('Contrast',  100, 93)
create_trackbar('Saturation', 100, 96)     # 饱和度
create_trackbar('Hue', 180, 6)      # 调节色调


state = b'A'
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11), (-1, -1))

num_init = 0
cross_flag = b'L'

element_count = {}

while True:
    flag, frame = cap.read()
    # 如果正确读帧
    if frame is None:
        print("The camera is not open")
        continue
    if flag is True:
        cv2.imshow('frame', frame)  # 没用处理过的图像的显示

        frame_line = frame.copy()   # 线的框架
        src = frame_line[:250, :]
        cv2.imshow('src', src)
        src_line = src.copy()
        edge_img = src.copy()  # dst_img画出红线位置

        '''
        模式B，进行数字识别，当识别到数字时候，进行巡线
        '''
        if state == b'A':
            A_flag = 1
            # frame_copy1 = frame.copy()  # 画出矩形的相片
            # frame_copy2 = frame.copy()  # 用于切割检测矩形
            # frame_copy3 = frame.copy()  # 画出匹配矩形及其准确度
            '''
            function: 数字识别的图像预处理
            '''
            # 转为灰度图
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 大津法处理阈值，背景反色，采用索引取出图像
            # 预留回调函数，手动调节阈值thresh
            ths_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # 形态学操作
            close_frame = cv2.morphologyEx(ths_frame, cv2.MORPH_CLOSE, rectKernel)
            '''
            function: 边缘检测和轮廓查找
            '''
            # 边缘检测
            frame_copy = frame.copy()  # 原图展示预测数字
            frame_copy_2 = frame.copy()

            # 自动阈值处理
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_ths = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # 闭操作
            img_closed = cv2.morphologyEx(frame_ths, cv2.MORPH_CLOSE, rectKernel)
            # canny边缘检测
            temp = imutils.auto_canny(img_closed)
            # 轮廓查找
            contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # 计算轮廓
            for index, c in enumerate(contours):
                area = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                if area > 1000:
                    img_out = cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    res_img = frame_copy_2[y:y + h, x:x + w]
                    # 图像预处理
                    img = cv2.resize(res_img, (256, 256))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    temp = img.reshape(1, -1)
                    # PCA降维
                    temp = loaded_pca.transform(temp)
                    y_new_pred = loaded_svm.predict(temp)
                    cv2.putText(frame_copy, f"Predict: {y_new_pred}", (int(x), int(y)),
                                 font, font_scale, show_color, font_thickness)

                    my_list_str = str(y_new_pred)  # 将列表转换为字符串
                    num_init = my_list_str.replace('[', '').replace(']', '')  # 去除方括号
                    print(num_init)
                    element_count[num_init] = element_count.get(num_init, 0) + 1

                    if max(element_count.values()) == 20:
                        max_value = max(element_count.values())
                        max_keys = [key for key, value in element_count.items() if value == max_value]
                        num_init = max_keys[0]
                        if num_init == '1':
                            print("{:o<14}".format('@~a&1#'))
                            # sendMessage("{:o<14}".format('@~a&1#'))
                            A_flag = 0
                        if num_init == '2':
                            print("{:o<14}".format('@~a&2#'))
                            # sendMessage("{:o<14}".format('@~a&2#'))
                            A_flag = 0

                    if max(element_count.values()) == 100:
                        max_value = max(element_count.values())
                        max_keys = [key for key, value in element_count.items() if value == max_value]
                        num_init = max_keys[0]

                        print("{:o<14}".format('@~b&'+str(num_init)+'#'))
                        # sendMessage("{:o<14}".format('@~b&'+str(num_init)+'#'))

                    if max(element_count.values()) == 200:
                        max_value = max(element_count.values())
                        max_keys = [key for key, value in element_count.items() if value == max_value]
                        num_init = max_keys[0]
                        print("{:o<14}".format('@~c&'+str(num_init)+'#'))
                        # sendMessage("{:o<14}".format('@~c&'+str(num_init)+'#'))
                        A_flag = 0

                            # for key in element_count:
                            #     element_count[key] = 0
                            # A_flag = 0
                            # cv2.destroyWindow('frame_copy')
                            # break
            cv2.imshow('frame_copy', frame_copy)
            if A_flag == 0:
                cv2.destroyWindow('frame_copy')
                state = b'B'

        '''
        巡线
        '''
        if state == b'B':
            '''偏转角度和距离中心位置多远距离'''
            angle = 0
            # 转化成hsv通道进行处理
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            # 双阈值二值化
            mask = cv2.inRange(hsv, color_lower, color_upper)
            # 闭运算
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # ksize=5,5
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=1)
            # cv2.imshow("mask", mask)

            # 处理偏移量,重新命名二值化图像
            src_thr = mask.copy()

            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            for cnt in contours:
                cv2.drawContours(edge_img, cnt, -1, (255, 0, 0), 2)
                # print('area:', cv2.contourArea(cnt))
            # 画出红线
            if True:
                if len(contours) > 0:
                    cnt = max(contours, key=cv2.contourArea)
                    # print('area:', cv2.contourArea(cnt))
                    (color_x, color_y), color_radius = cv2.minEnclosingCircle(cnt)
                    # print(color_x, color_y, color_radius)
                    if color_radius > 5:
                        # 将检测到的颜色用原形线圈标记出来
                        cv2.circle(frame, (int(color_x), int(color_y)), int(color_radius), (0, 0, 0), 10)

            # cv2.imshow("edge_img", edge_img)
            # cv2.imshow("circle_frame", frame)


            print(src_thr.shape)
            '''
            图片的大小为 250 * 640
            '''
            # 看图片的底部
            color = src_thr[240]
            # 再看第200行的像素值与第300行的像素值
            color1 = src_thr[60]
            color2 = src_thr[120]

            '''
            寻找中心点并计算偏移量
            '''
            white_index = np.where(color == 255)
            white_count_bottle = np.sum(color == 255)
            # print("white_index", white_index[0])
            if len(white_index[0]) > 1:

                # 防止white_count=0的报错
                if white_count_bottle == 0:
                    white_count_bottle = 1

                '''
                计算偏移量
                '''
                # 在计算偏移的角度。
                black_count1_judge = np.sum(color1 == 0)  # 计算黑色像素的个数，如果黑色像素太多则没用线
                black_count2_judge = np.sum(color2 == 0)

                white_index1 = np.where(color1 == 255)
                white_index2 = np.where(color2 == 255)
                white_count1 = np.sum(color1 == 255)
                white_count2 = np.sum(color2 == 255)

                '''
                计算偏移角度
                '''

                if white_count1 > 1 and white_count2 > 1:
                    print("start trace red line")
                    center1 = (white_index1[0][white_count1 - 1] + white_index1[0][0]) / 2
                    direction1 = center1 - 320
                    center2 = (white_index2[0][white_count2 - 1] + white_index2[0][0]) / 2
                    direction2 = center2 - 320
                    print("white_center1:", center1, "white_center2:", center2)
                    angle = '%.2f' % (math.degrees(np.arctan(30 / (direction2 - direction1))))  # 计算k
                    print("偏转角为：", angle)
                    angle = int(float(angle))
                    cv2.line(src_line, (int(center1), 60), (int(center2), 120), color=(255, 0, 0),
                             thickness=5)  # 蓝色的线
                    cv2.line(src_line, (0, 60), (640, 60), color=(0, 0, 255), thickness=3)  # 红色的线
                    cv2.line(src_line, (0, 120), (640, 120), color=(0, 0, 255), thickness=3)
                    pass

                else:  # 如果没有发现第150行喝第300行的黑线
                    angle = 0
                    print('偏转角为：ERROR')
                    pass

                # 找到白色像素的中心点位置
                center = (white_index[0][white_count_bottle - 1] + white_index[0][0]) / 2
                print('白色像素的中心点位置', center)
                direction = center - 320  # 在实际操作中，我发现当黑线处于小车车体正中央的时候应该减去320
                direction = int('%4d' % direction)

                if direction > 0:
                    print("{:o<14}".format('@^R&' + str(angle) + '|' + str(direction) + '#'))
                    # sendMessage("{:o<14}".format('@^R&'+str(angle)+'|'+str(direction)+'#'))

                if direction < 0:
                    direction = abs(direction)
                    print("{:o<14}".format('@^R&' + str(angle) + '|' + str(direction) + '#'))
                    # sendMessage("{:o<14}".format('@^L&'+str(angle)+'|'+str(direction)+'#'))
                # '''
                # 计算出center与标准中心点的偏移量
                # 当红线处于小车车体右侧的时候，偏移量为正值，黑线处于小车车体左侧的时候，偏移量为负值（处于小车视角）
                # 计算轮子的速度，使用差速偏移
                # '''
                # if direction > 0:
                #     right_param = 1999 + (direction * 4)  # 这个参数可以后期更改
                #     light_param = 1999
                #     final_param = 'r:' + str(light_param) + 'l:' + str(right_param) + '\r\n'
                #     print(final_param)
                #     # time.sleep(0.2)
                #     # ser.write(final_param.encode())
                # else:
                #     media = -direction
                #     light_param = 1999 + (media * 4)
                #     right_param = 1999
                #     final_param = 'r:' + str(light_param) + 'l:' + str(right_param) + '\r\n'
                #     print(final_param)
                #     # time.sleep(0.2)
                #     # ser.write(final_param.encode())


            else:
                print("failed to find index")
            cv2.imshow("src_line", src_line)
            # 寻找十字点
            color_ten = src_thr[120]
            # 找到白色的像素点个数
            white_count = np.sum(color_ten == 255)
            # cv2.imshow('frame_line', frame_line)
            # print("白色十字像素点为：", white_count)
            if white_count >= 300:  # 假如识别到了白色十字，具体的值还需要调试
                print("识别到白色的十字")
                state = b'C'
                cv2.destroyWindow('frame_line')
                cv2.destroyWindow('mask')
                cv2.destroyWindow('src_line')
                cv2.destroyWindow("edge_img")
                cv2.destroyWindow("circle_frame")

            else:
                print("没有识别到白色十字")



        if state == b'C':
            frame_cross_copy = frame.copy()

            frame_crop_2 = frame.copy()
            frame_left = frame[:, :320]
            frame_right = frame[:, 320:]
            # cv2.imshow('frame_left', frame_left)
            # cv2.imshow('frame_right', frame_right)
            left_pred = None
            right_pred = None

            if cross_flag == b'L':
                print('L')
                '''
                function: 边缘检测和轮廓查找
                '''
                # 边缘检测
                frame_copy_2 = frame_left.copy()
                # 自动阈值处理
                frame_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
                frame_ths = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # 闭操作
                img_closed = cv2.morphologyEx(frame_ths, cv2.MORPH_CLOSE, rectKernel)
                # canny边缘检测
                temp = imutils.auto_canny(img_closed)
                # 轮廓查找
                contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                print(len(contours))
                if len(contours) > 0:
                    # 计算轮廓
                    for index, c in enumerate(contours):
                        area = cv2.contourArea(c)
                        x, y, w, h = cv2.boundingRect(c)
                        if area > 500:
                            img_out = cv2.rectangle(frame_cross_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            res_img = frame_copy_2[y:y + h, x:x + w]
                            # 图像预处理
                            img = cv2.resize(res_img, (256, 256))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            temp = img.reshape(1, -1)
                            # PCA降维
                            temp = loaded_pca.transform(temp)
                            y_new_pred = loaded_svm.predict(temp)
                            print(y_new_pred)
                            cv2.putText(frame_cross_copy, f"Predict: {y_new_pred}", (int(x), int(y)),
                                        font, font_scale, show_color, font_thickness)

                            num_value = y_new_pred.item() if len(y_new_pred) == 1 else y_new_pred[0]
                            arr = set()
                            arr.add(num_value)
                            print(len(arr))
                            if len(arr) > 0:
                                for i in arr:
                                    print(num_init, i)
                                    if int(i) == int(num_init):
                                        left_pred = i
                                        print(left_pred)
                                        print("left")
                                        break
                                    else:
                                        print('TO R')
                                        cross_flag = b'R'
                            else:
                                print('TO R')
                                cross_flag = b'R'
                        else:
                            print('TO R')
                            cross_flag = b'R'
                else:
                    print('TO R')
                    cross_flag = b'R'

            if cross_flag == b'R':
                print('R')
                '''
                function: 边缘检测和轮廓查找
                '''
                # 边缘检测
                frame_copy_2 = frame_right.copy()
                # 自动阈值处理
                frame_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
                frame_ths = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # 闭操作
                img_closed = cv2.morphologyEx(frame_ths, cv2.MORPH_CLOSE, rectKernel)
                # canny边缘检测
                temp = imutils.auto_canny(img_closed)
                # 轮廓查找
                contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                print(len(contours))
                if len(contours) > 0:
                    # 计算轮廓
                    for index, c in enumerate(contours):
                        area = cv2.contourArea(c)
                        x, y, w, h = cv2.boundingRect(c)
                        if area > 500:
                            img_out = cv2.rectangle(frame_cross_copy, (x + 320, y), (x + w + 320, y + h), (0, 0, 255), 2)
                            res_img = frame_copy_2[y:y + h, x:x + w]
                            # 图像预处理
                            img = cv2.resize(res_img, (256, 256))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            temp = img.reshape(1, -1)
                            # PCA降维
                            temp = loaded_pca.transform(temp)
                            y_new_pred = loaded_svm.predict(temp)
                            print(y_new_pred)
                            cv2.putText(frame_cross_copy, f"Predict: {y_new_pred}", (int(x + 320), int(y)),
                                        font, font_scale, show_color, font_thickness)

                            num_value = y_new_pred.item() if len(y_new_pred) == 1 else y_new_pred[0]
                            arr = set()
                            arr.add(num_value)
                            if len(arr) > 0:
                                for i in arr:
                                    print(num_init, i)
                                    if int(i) == int(num_init):
                                        right_pred = i
                                        print(right_pred)
                                        print("right")
                                        break
                                    else:
                                        print('TO L')
                                        cross_flag = b'L:'
                            else:
                                print('TO L')
                                cross_flag = b'L'
                        else:
                            print('TO L')
                            cross_flag = b'L'
                else:
                    print('TO L')
                    cross_flag = b'L'

            cv2.imshow('frame_cross_copy_predict', frame_cross_copy)

            if left_pred == int(num_init):
                print("{:o<14}".format('@*L#'))

                state = b'B'  # 左侧识别到目标数字，状态变为b'B'
            elif right_pred == int(num_init):
                print("{:o<14}".format('@*R#'))
            else:
                print("Target number not recognized on either side.")
                state = b'C'
                # 没有识别到目标数字，保持当前状态不变





    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        break