from pynput.keyboard import Key, Listener
import cv2
import imutils


# 按键获取数据集
global res_img
counts = 1    # 图片计数
def on_press(key):
    global counts
    # 回车触发
    if key == Key.enter:
        # 保存图片路径，制作单一数据集
        cv2.imwrite(f"Template/{counts}.jpg", res_img)
        print(f"Save sucess {counts}")
        counts += 1



# 开启键盘监听
listener = Listener(on_press=on_press)
listener.start()


cap = cv2.VideoCapture(1)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11))
while True:
    flag, frame = cap.read()
    if frame is None:
        continue
    if flag is True:
        frame_copy = frame.copy()
        frame_copy1 = frame.copy()
        frame_copy2 = frame.copy()
        # 转为灰度图
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 大津法处理阈值，背景反色，采用索引取出图像
        # 预留回调函数，手动调节阈值thresh
        ths_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # 形态学操作
        img_closed = cv2.morphologyEx(ths_frame, cv2.MORPH_CLOSE, rectKernel)
        # 边缘检测
        img_blur = cv2.GaussianBlur(img_closed, (3, 3), 0)
        img_temp = cv2.Canny(img_blur, 100, 200)
        # 轮廓查找
        contours, hierarchy = cv2.findContours(img_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for region in contours:
            area = cv2.contourArea(region)
            x, y, w, h = cv2.boundingRect(region)
            if area > 10000:
                img_out = cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # 获取轮廓区域的图像
                res_img = frame_copy2[y:y + h, x:x + w]

                cv2.putText(frame_copy, f'Area: {area:.2f}', (x, y - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                # 显示数据图像
                cv2.imshow("res_img", res_img)

        cv2.imshow("find the contours", frame_copy)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            listener.stop()
            break

