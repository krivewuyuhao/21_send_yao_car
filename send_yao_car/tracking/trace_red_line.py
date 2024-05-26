import math
import cv2
import numpy as np

color_x = color_y = color_radius = 0

'''
转化到hsv空间下处理红色的阈值
'''
color_hsv = {"red": ((0, 100, 100), (10, 255, 255))}   # 红色像素在hsv空间下的阈值
color = "red"
color_lower = np.array([color_hsv[color][0][0], color_hsv[color][0][1], color_hsv[color][0][2]])
color_upper = np.array([color_hsv[color][1][0], color_hsv[color][1][1], color_hsv[color][1][2]])

image = cv2.imread("test.jpg")
image = cv2.resize(image, (53, 91))
dst_img = image

print(image.shape)
cv2.imshow("src", image)

# # 如果直接二值化的效果
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_BINARY_INV)[1]
# cv2.imshow('binary_image', binary_image)


def Color_Recongnize():
    global color_lower, color_upper
    global color_x, target_servox, picture

    # 高斯滤波处理
    frame = image.copy()

    # 转化成hsv通道进行处理
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 双阈值二值化
    mask = cv2.inRange(hsv, color_lower, color_upper)
    # 闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # ksize=5,5
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=1)
    cv2.imshow("mask", mask)

    # 处理偏移量,重新命名二值化图像
    src_thr = mask.copy()
    src_line = image.copy()


    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    for cnt in contours:
        cv2.drawContours(dst_img, cnt, -1, (255, 0, 0), 2)
        print('area:', cv2.contourArea(cnt))
    # 画出红线
    if True:
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            print('area:', cv2.contourArea(cnt))
            (color_x, color_y), color_radius = cv2.minEnclosingCircle(cnt)
            print(color_x, color_y, color_radius)
            if color_radius > 5:
                # 将检测到的颜色用原形线圈标记出来
                cv2.circle(frame, (int(color_x), int(color_y)), int(color_radius), (0, 0, 0), 10)

    cv2.imshow("dst_img", dst_img)
    cv2.imshow("circle_frame", frame)

    print(src_thr.shape)
    src_thr = cv2.resize(src_thr, (53, 91))
    # 看图片的底部
    color = src_thr[90]
    # 再看第200行的像素值与第300行的像素值
    color1 = src_thr[30]
    color2 = src_thr[60]

    # 找到白色的像素点个数
    white_count = np.sum(color == 255)
    print("白色像素点为：", white_count)

    if white_count >= 300:  # 假如识别到了白色十字，具体的值还需要调试
        print("识别到白色的十字")
    else:
        print("没有识别到白色十字")

    white_index = np.where(color == 255)
    print("white_index", white_index[0])
    # 防止white_count=0的报错
    if white_count == 0:
        white_count = 1
    # 在这里，我们要计算偏移的角度。
    black_count1_judge = np.sum(color1 == 0)  # 第200行如果全是黑色的话就不计算角度了
    black_count2_judge = np.sum(color2 == 0)
    white_index1 = np.where(color1 == 255)
    white_index2 = np.where(color2 == 255)
    white_count1 = np.sum(color1 == 255)
    white_count2 = np.sum(color2 == 255)

    if black_count1_judge < 50 and black_count2_judge < 50:
        center1 = (white_index1[0][white_count1 - 1] + white_index1[0][0]) / 2
        direction1 = center1 - 45
        center2 = (white_index2[0][white_count2 - 1] + white_index2[0][0]) / 2
        direction2 = center2 - 45
        print("white_center1:", center1, "white_center2:", center2)
        angle = '%.2f' % (math.degrees(np.arctan(30 / (direction2 - direction1))))  # 计算k
        print("偏转角为：", angle)
        cv2.line(src_line, (int(center1), 30), (int(center2), 60), color=(255, 0, 0), thickness=3)  # 蓝色的线
        cv2.line(src_line, (0, 30), (53, 30), color=(0, 0, 255), thickness=3)  # 红色的线
        cv2.line(src_line, (0, 60), (53, 60), color=(0, 0, 255), thickness=3)
        cv2.imshow("src_line", src_line)
        pass
    if black_count1_judge >= 630 or black_count2_judge >= 630:  # 如果没有发现第150行喝第300行的黑线
        # angle = ERROR
        print('偏转角为：ERROR')
        pass
    # 找到白色像素的中心点位置
    center = (white_index[0][white_count - 1] + white_index[0][0]) / 2
    direction = center - 45  # 在实际操作中，我发现当黑线处于小车车体正中央的时候应该减去302
    direction = int('%4d' % direction)
    print("方向为：", direction)
    # 计算出center与标准中心点的偏移量
    '''当红线处于小车车体右侧的时候，偏移量为正值，黑线处于小车车体左侧的时候，偏移量为负值（处于小车视角）'''
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




    try:
        # 找到二值化后白色的像素点个数
        white_count = np.sum(color == 255)
        # 找到二值化后白色的像素点索引
        white_index = np.where(color == 255)
        # 防止white_count=0的报错
        if white_count == 0:
            white_count = 1
        # 计算方法应该是边缘检测，计算白色边缘的位置和/2，即是白色的中央位置。
        center = (white_index[0][white_count - 1] + white_index[0][0]) / 2
        # 计算出center与标准中心点的偏移量，因为图像大小是640，因此标准中心是320，因此320不能改。
        direction = center - 20
        print(direction)

    except:
        # continue
        print("no find")


# rows, cols = frame.shape[:2]
#     cv2.imshow('mark', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Color_Recongnize()

