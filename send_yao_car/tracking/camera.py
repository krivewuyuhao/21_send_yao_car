import math

import cv2
import numpy as np

'''
function: 创建回调函数
'''


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


# 创建滑动条的函数
# 创建滑动条的函数
def create_trackbar(name, max_range, default_value):
    cv2.createTrackbar(name, "frame", default_value, max_range, lambda val: on_trackbar(val, name))


'''
转化到hsv空间下处理红色的阈值,具体怎么处理反光问题？该看看现场环境
'''
color_x = color_y = color_radius = 0
color_hsv = {"red": ((0, 100, 100), (10, 255, 255))}   # 红色像素在hsv空间下的阈值范围
color = "red"
color_lower = np.array([color_hsv[color][0][0], color_hsv[color][0][1], color_hsv[color][0][2]])
color_upper = np.array([color_hsv[color][1][0], color_hsv[color][1][1], color_hsv[color][1][2]])

# 创建视频捕获对象
cap = cv2.VideoCapture(1)
print('高度:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('宽度:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))

'''创建一个窗口，用交互式方式调节摄像头'''
cv2.namedWindow("frame")
# 创建滑动条并关联回调函数
create_trackbar('Brightness',  100, 59)
create_trackbar('Contrast',  100, 93)
create_trackbar('Saturation', 100, 96)     # 饱和度
create_trackbar('Hue', 180, 6)      # 调节色调
'''
frame 是最原始的图像
'''
while True:
    flag, frame = cap.read()
    src = frame[:250, :]        # 处理该部分图像，对之后的图像有好处，作为原始图像进行处理
    cv2.imshow('src', src)

    if frame is None:
        print("not find camera")

    if flag is True:
        cv2.imshow("frame", frame)

        # 转化成hsv通道进行处理
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        # 双阈值二值化
        mask = cv2.inRange(hsv, color_lower, color_upper)
        # 闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # ksize=5,5
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=1)
        cv2.imshow("mask", mask)

        # 处理偏移量,重新命名二值化图像
        src_thr = mask.copy()
        src_line = src.copy()
        edge_img = src.copy()  # dst_img画出红线位置

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

        cv2.imshow("edge_img", edge_img)
        cv2.imshow("circle_frame", frame)

        print(src_thr.shape)
        '''
        图片的大小为 250 * 640
        '''
        # 看图片的底部
        color = src_thr[240]
        # 再看第200行的像素值与第300行的像素值
        color1 = src_thr[60]
        color2 = src_thr[120]

        # 寻找十字点
        color_ten = src_thr[120]

        # 找到白色的像素点个数
        white_count = np.sum(color_ten == 255)
        # print("白色十字像素点为：", white_count)
        if white_count >= 300:  # 假如识别到了白色十字，具体的值还需要调试
            print("识别到白色的十字")
        else:
            print("没有识别到白色十字")

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
                cv2.line(src_line, (int(center1), 60), (int(center2), 120), color=(255, 0, 0), thickness=5)  # 蓝色的线
                cv2.line(src_line, (0, 60), (640, 60), color=(0, 0, 255), thickness=3)  # 红色的线
                cv2.line(src_line, (0, 120), (640, 120), color=(0, 0, 255), thickness=3)
                cv2.imshow("src_line", src_line)
                pass

            else:  # 如果没有发现第150行喝第300行的黑线
                # angle = ERROR
                print('偏转角为：ERROR')
                pass

            # 找到白色像素的中心点位置
            center = (white_index[0][white_count_bottle - 1] + white_index[0][0]) / 2
            print(center)
            direction = center - 320  # 在实际操作中，我发现当黑线处于小车车体正中央的时候应该减去320
            direction = int('%4d' % direction)
            print("方向为：", direction)
            '''
            计算出center与标准中心点的偏移量
            当红线处于小车车体右侧的时候，偏移量为正值，黑线处于小车车体左侧的时候，偏移量为负值（处于小车视角）
            计算轮子的速度，使用差速偏移
            '''
            if direction > 0:
                right_param = 1999 + (direction * 4)  # 这个参数可以后期更改
                light_param = 1999
                final_param = 'r:' + str(light_param) + 'l:' + str(right_param) + '\r\n'
                print(final_param)
                # time.sleep(0.2)
                # ser.write(final_param.encode())
            else:
                media = -direction
                light_param = 1999 + (media * 4)
                right_param = 1999
                final_param = 'r:' + str(light_param) + 'l:' + str(right_param) + '\r\n'
                print(final_param)
                # time.sleep(0.2)
                # ser.write(final_param.encode())
        else:
            print("failed to find index")

    # 按下ESC键退出循环
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放视频捕获对象和窗口
cap.release()
cv2.destroyAllWindows()