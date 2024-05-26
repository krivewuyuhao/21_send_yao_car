import array
import math
import imutils
import joblib
import cv2
import numpy as np
import time
import serial

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
ser = serial.Serial(port="/dev/ttyAMA0", baudrate=115200, timeout=0)
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
loaded_svm = joblib.load('best_svm_model.joblib')
loaded_pca = joblib.load('best_pca_model.joblib')
last_time = time.time()
'''
转化到hsv空间下处理红色的阈值,具体怎么处理反光问题？该看看现场环境
'''
color_x = color_y = color_radius = 0
color_hsv = {"red": ((0, 100, 100), (10, 255, 255)),
             "black": ((0, 0, 0), (180, 255, 180))
             }   # 红色像素在hsv空间下的阈值范围

color = ["red", "black"]
red_lower = np.array([color_hsv[color[0]][0][0], color_hsv[color[0]][0][1], color_hsv[color[0]][0][2]])
red_upper = np.array([color_hsv[color[0]][1][0], color_hsv[color[0]][1][1], color_hsv[color[0]][1][2]])
print(red_lower)

black_lower = np.array([color_hsv[color[1]][0][0], color_hsv[color[1]][0][1], color_hsv[color[1]][0][2]])
black_upper = np.array([color_hsv[color[1]][1][0], color_hsv[color[1]][1][1], color_hsv[color[1]][1][2]])

'''创建一个窗口，用交互式方式调节摄像头'''
cv2.namedWindow("src")
# 创建滑动条并关联回调函数
create_trackbar('Brightness',  100, 59)
create_trackbar('Contrast',  100, 93)
create_trackbar('Saturation', 100, 96)     # 饱和度
create_trackbar('Hue', 180, 6)      # 调节色调

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11), (-1, -1))

'''
状态的初始化
'''
state = b'B'
num_init = 0
cross_flag = b'L'
element_count = {}

'''
鉴于树莓派的图像处理
只显示原始图像，二值化的图像，处理完画线的图像
'''

'''
树莓派的性能太差，所以处理图像功能与自己的电脑不同这里只能降低图像的分辨率
设置降低的倍数
'''
'''
定义降低多少像素值及裁剪大小，具体看小车
'''
multiple = 2
crop_size = 0.6
while True:
    flag, frame = cap.read()
    # 如果正确读帧
    if frame is None:
        print("The camera is not open")
        continue
    if flag is True:
        width = frame.shape[1]
        height = frame.shape[0]
        frame = cv2.resize(frame, (int(height * crop_size), int(width//multiple)))
        frame_line = frame.copy()   # 线的框架

        src = frame_line[:int(frame.shape[0]//multiple), :]
        src_width = src.shape[1]
        src_height = src.shape[0]
        # print(src_width, src_height)
        
        cv2.imshow('src', src)

        '''
        模式B，进行数字识别，当识别到数字时候，进行巡线
        '''
        if state == b'A':
            '''
            function: 数字识别的图像预处理
            '''
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 自动阈值处理
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
                if area > 1500:
                    img_out = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    res_img = frame[y:y + h, x:x + w]
                    # 图像预处理
                    '''
                    确保图像预处理与模型预处理时候一致
                    '''
                    img = cv2.resize(res_img, (128, 128))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    temp = img.reshape(1, -1)
                    # PCA降维
                    temp = loaded_pca.transform(temp)
                    y_new_pred = loaded_svm.predict(temp)
                    cv2.putText(frame, f"Predict: {y_new_pred}", (int(x), int(y)),
                                 font, 0.5, show_color, font_thickness)
                    my_list_str = str(y_new_pred)  # 将列表转换为字符串
                    num_init = my_list_str.replace('[', '').replace(']', '')  # 去除方括号
                    # print(num_init)
                    element_count[num_init] = element_count.get(num_init, 0) + 1
                    if max(element_count.values()) == 20:
                        max_value = max(element_count.values())
                        max_keys = [key for key, value in element_count.items() if value == max_value]
                        num_init = max_keys[0]
                        if num_init == '1':
                            print("{:o<14}".format('@~a&1#'))
                            sendMessage("{:o<14}".format('@~a&1#'))

                        if num_init == '2':
                            print("{:o<14}".format('@~a&2#'))
                            sendMessage("{:o<14}".format('@~a&2#'))

                    if max(element_count.values()) == 100:
                        max_value = max(element_count.values())
                        max_keys = [key for key, value in element_count.items() if value == max_value]
                        num_init = max_keys[0]
                        print("{:o<14}".format('@~b&'+str(num_init)+'#'))
                        sendMessage("{:o<14}".format('@~b&'+str(num_init)+'#'))
                    if max(element_count.values()) == 200:
                        max_value = max(element_count.values())
                        max_keys = [key for key, value in element_count.items() if value == max_value]
                        num_init = max_keys[0]
                        print("{:o<14}".format('@~c&'+str(num_init)+'#'))
                        sendMessage("{:o<14}".format('@~c&'+str(num_init)+'#'))
                        
            temp = ser.read(1)
            ser.flushInput()
            if temp != b'':
                print(temp)
            if temp == b'A' or temp == b'B' or temp == b'C' :
                state = temp
            # cv2.imshow('frame_copy', frame_copy)
            '''
            读取单片机数据切换模式
            '''
            A_flag = 1
            if A_flag == 0:
                # cv2.destroyWindow('frame_copy')
                state = b'B'
        '''
        功能2实现巡线
        '''
        if state == b'B':
            '''偏转角度和距离中心位置多远距离'''
            angle = 0
            # 转化成hsv通道进行处理
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            # 双阈值二值化
            mask = cv2.inRange(hsv, red_lower, red_upper)
            # 闭运算
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # ksize=5,5
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=1)
            cv2.imshow("mask", mask)

            # 处理偏移量,重新命名二值化图像
            src_thr = mask.copy()

            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            # for cnt in contours:
            #     cv2.drawContours(frame, cnt, -1, (255, 0, 0), 2)
            #     # print('area:', cv2.contourArea(cnt))
            # # 画出红线
            # if True:
            #     if len(contours) > 0:
            #         cnt = max(contours, key=cv2.contourArea)
            #         # print('area:', cv2.contourArea(cnt))
            #         (color_x, color_y), color_radius = cv2.minEnclosingCircle(cnt)
            #         # print(color_x, color_y, color_radius)
            #         if color_radius > 5:
            #             # 将检测到的颜色用原形线圈标记出来
            #             cv2.circle(frame, (int(color_x), int(color_y)), int(color_radius), (0, 0, 0), 10)
            #
            # # cv2.imshow("edge_img", frame)
            # # cv2.imshow("circle_frame", frame)
            '''
            图片的大小为 250 * 640
            '''
            # 看图片的底部
            
            color = src_thr[src_height-20]
            # 再看第200行的像素值与第300行的像素值
            line_up = int(src_height * 0.2)
            line_down = int(src_height * 0.8)
            color1 = src_thr[line_up]
            color2 = src_thr[line_down]
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
                    # print("start trace red line")
                    center1 = (white_index1[0][white_count1 - 1] + white_index1[0][0]) / 2
                    direction1 = center1 - src_width//2
                    center2 = (white_index2[0][white_count2 - 1] + white_index2[0][0]) / 2
                    direction2 = center2 - src_width//2
                    # print("white_center1:", center1, "white_center2:", center2)
                    angle = '%.2f' % (math.degrees(np.arctan((direction2 - direction1)/(line_down - line_up))))  # 计算k
                    cv2.putText(frame, f"angle: {angle}", (5, 70),
                                font, font_scale, (0, 0, 255), font_thickness)
                    angle = int(float(angle))
                    cv2.line(frame, (int(center1), line_up), (int(center2), line_down), color=(255, 0, 0),
                             thickness=5)  # 蓝色的线
                    cv2.line(frame, (0, line_up), (640, line_up), color=(0, 0, 255), thickness=3)  # 红色的线
                    cv2.line(frame, (0, line_down), (640, line_down), color=(0, 0, 255), thickness=3)
                    pass
                else:  # 如果没有发现第150行喝第300行的黑线
                    angle = 0
                    cv2.putText(frame, f"angle: ERROR", (5, 70),
                                font, font_scale, (0, 0, 255), font_thickness)
                    pass
                
                # cv2.imshow("src_line", src_line)
                # 找到白色像素的中心点位置
                center = (white_index[0][white_count_bottle - 1] + white_index[0][0]) / 2
                # print('白色像素的中心点位置', center)
                radius = center - src_width // 2  # 在实际操作中，我发现当黑线处于小车车体正中央的时候应该减去320
                radius = int('%4d' % radius)
                cv2.putText(frame, f"radius: {radius}", (5, 110),
                            font, font_scale, (0, 0, 255), font_thickness)
                angle =abs(angle)
                if radius >= 0:
                    # print("{:o<14}".format('@^R&' + str(angle) + '|' + str(radius) + '#'))
                    sendMessage("{:o<14}".format('@^R&'+str(angle)+'|'+str(radius)+'#'))
                    # time.sleep(0.1)
                if radius < 0:
                    radius = abs(radius)
                    # print("{:o<14}".format('@^L&' + str(angle) + '|' + str(radius) + '#'))
                    sendMessage("{:o<14}".format('@^L&'+str(angle)+'|'+str(radius)+'#'))
                    # time.sleep(0.1)
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
                cv2.putText(frame, f"failed to find index", (5, 70),
                            font, 0.8, (0, 0, 255), font_thickness)

            # 寻找十字点
            color_ten = src_thr[line_up]
            # 找到白色的像素点个数
            white_count = np.sum(color_ten == 255)
            # cv2.imshow('frame_line', frame_line)
            # print("白色十字像素点为：", white_count)
            if white_count >= 300:  # 假如识别到了白色十字，具体的值还需要调试
                cv2.putText(frame, f"find cross line", (5, 110),
                            font, font_scale, (0, 0, 255), font_thickness)
                if num_init == '1':
                    sendMessage("{:o<14}".format('@*L#'))
                elif num_init == '2':
                    sendMessage('@*R#')
                else:
                    state = b'C'

                # cv2.destroyWindow('frame_line')
                # cv2.destroyWindow('mask')
                # cv2.destroyWindow('src_line')
                # cv2.destroyWindow("edge_img")
                # cv2.destroyWindow("circle_frame")
            else:
                cv2.putText(frame, f"failed to find cross line", (5, 140),
                            font, 0.7, (0, 0, 255), font_thickness)

            '''
            寻找黑色框框，作为停止的标志
            '''
            # 创建掩膜，提取黑色区域
            mask_black = cv2.inRange(hsv, black_lower, black_upper)
            
            mask_black = mask_black[ : ,30:230]

            cv2.imshow("mask_black", mask_black)
            # 寻找轮廓
            contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 遍历轮廓，筛选出近似矩形的轮廓
            rectangles = []
            for contour in contours_black:
                # 计算轮廓的近似多边形
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                # 如果多边形的顶点数为4，且它是凸四边形，则认为是矩形
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    # 计算矩形的面积
                    area = cv2.contourArea(contour)
                    # 可以设置一个面积阈值来过滤小的矩形
                    if area > 500:  # 根据实际情况调整
                        rectangles.append(contour)
            # 画出识别到的矩形框
            for rectangle in rectangles:
                (x, y, w, h) = cv2.boundingRect(rectangle)
                x = x+30
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(rectangles) > 2:
                # 发送停止信号
                cv2.putText(frame, f"stop", (5, 160),
                            font, 0.8, (0, 0, 255), font_thickness)


            
            temp = ser.read(1)
            ser.flushInput()
            if temp != b'':
                print(temp)
            if temp == b'A' or temp == b'B' or temp == b'C' :
                state = temp
                cv2.destroyWindow('mask')

        if state == b'C':
            src_left = src[:, :src_width // 2]
            src_right = src[:, src_width // 2:]
            cv2.imshow('frame_left', src_left)
            cv2.imshow('frame_right', src_right)
            left_pred = None
            right_pred = None

            if cross_flag == b'L':
                '''
                function: 边缘检测和轮廓查找
                '''
                # 边缘检测
                frame_copy_2 = src_left.copy()
                # 自动阈值处理
                frame_gray = cv2.cvtColor(src_left, cv2.COLOR_BGR2GRAY)
                frame_ths = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # 闭操作
                img_closed = cv2.morphologyEx(frame_ths, cv2.MORPH_CLOSE, rectKernel)
                # canny边缘检测
                temp = imutils.auto_canny(img_closed)
                # 轮廓查找
                contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    # 计算轮廓
                    for index, c in enumerate(contours):
                        area = cv2.contourArea(c)
                        x, y, w, h = cv2.boundingRect(c)
                        if area > 1000:
                            img_out = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            res_img = frame_copy_2[y:y + h, x:x + w]
                            # 图像预处理
                            img = cv2.resize(res_img, (128, 128))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            temp = img.reshape(1, -1)
                            # PCA降维
                            temp = loaded_pca.transform(temp)
                            y_new_pred = loaded_svm.predict(temp)
                            cv2.putText(frame, f"Predict: {y_new_pred}", (int(x), int(y)),
                                        font, font_scale, show_color, font_thickness)

                            num_value = y_new_pred.item() if len(y_new_pred) == 1 else y_new_pred[0]
                            arr = set()
                            arr.add(num_value)
                            print(len(arr))
                            if len(arr) > 0:
                                for i in arr:
                                    print(num_init, i)
                                    if int(i) == int(num_init):
                                        cv2.putText(frame, f"Left", (5, 70),
                                                    font, font_scale, (0, 0, 255), font_thickness)
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

                '''
                function: 边缘检测和轮廓查找
                '''
                # 边缘检测
                frame_copy_2 = src_right.copy()
                # 自动阈值处理
                frame_gray = cv2.cvtColor(src_right, cv2.COLOR_BGR2GRAY)
                frame_ths = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # 闭操作
                img_closed = cv2.morphologyEx(frame_ths, cv2.MORPH_CLOSE, rectKernel)
                # canny边缘检测
                temp = imutils.auto_canny(img_closed)
                # 轮廓查找
                contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # print(len(contours))
                if len(contours) > 0:
                    # 计算轮廓
                    for index, c in enumerate(contours):
                        area = cv2.contourArea(c)
                        x, y, w, h = cv2.boundingRect(c)
                        if area > 1000:
                            img_out = cv2.rectangle(frame, (x + src_width//2, y), (x + w + src_width//2, y + h), (0, 0, 255), 2)
                            res_img = frame_copy_2[y:y + h, x:x + w]
                            # 图像预处理
                            img = cv2.resize(res_img, (128, 128))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            temp = img.reshape(1, -1)
                            # PCA降维
                            temp = loaded_pca.transform(temp)
                            y_new_pred = loaded_svm.predict(temp)
                            # print(y_new_pred)
                            cv2.putText(frame, f"Predict: {y_new_pred}", (int(x + 320), int(y)),
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
                                        cv2.putText(frame, f"Right", (5, 70),
                                                    font, font_scale, (0, 0, 255), font_thickness)
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

            # cv2.imshow('frame_cross_copy_predict', frame_cross_copy)

            if left_pred == int(num_init):
                print("{:o<14}".format('@*L#'))
                sendMessage("{:o<14}".format('@*L#'))
                state = b'B'  # 左侧识别到目标数字，状态变为b'B'

            elif right_pred == int(num_init):
                print("{:o<14}".format('@*R#'))
                sendMessage("{:o<14}".format('@*R#'))
            else:
                cv2.putText(frame, f"not recognized", (5, 110),
                            font, font_scale, (0, 0, 255), font_thickness)
                state = b'C'
                sendMessage('@=m#')
                # 没有识别到目标数字，保持当前状态不变
                
            temp = ser.read(1)
            ser.flushInput()
            if temp != b'':
                print(temp)
            if temp == b'A' or temp == b'B' or temp == b'C' :
                state = temp
                cv2.destroyWindow('frame_left')
                cv2.destroyWindow('frame_right')

        # 获取当前时间
        current_time = time.time()
        # 计算自上次迭代以来经过的时间（以秒为单位）
        elapsed_time = current_time - last_time
        last_time = current_time
        # 计算帧率，每秒的帧数（FPS）
        fps = 1.0 / elapsed_time if elapsed_time > 0 else float('inf')
        # 绘制帧率信息
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), font, font_scale, (0, 255, 0), font_thickness)
        cv2.imshow('frame', frame)  # 图像的显示
        
    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
