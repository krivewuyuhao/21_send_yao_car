import cv2
import numpy as np
# 读取所有的模板图像
'''
function: 模板的加载
'''
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11), (-1, -1))

template = []
for i in range(1, 9):
    template_path = f'Template/{i}.jpg'
    each_template = cv2.imread(template_path, 0)
    if each_template is None:
        print(f"Cannot read template image: {template_path}")
        continue
    thr_template = cv2.threshold(each_template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    print("src_shape:", each_template.shape)
    template.append(thr_template)   # 模板是二值化后的图像
    # template.append(each_template)  # 模板是二值化后的图像
print(f"Number of templates loaded: {len(template)}")


cap = cv2.VideoCapture(1)

# 核的大小


while True:
    flag, frame = cap.read()
    # 如果正确读帧
    if frame is None:
        print("The camera is not open")
        continue
    if flag is True:
        frame_copy1 = frame.copy()  # 画出矩形的相片
        frame_copy2 = frame.copy()  # 用于切割检测矩形
        frame_copy3 = frame.copy()  # 画出匹配矩形及其准确度

        '''
        function: 图像预处理
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
        blur_img = cv2.GaussianBlur(close_frame, (3, 3), 0)
        canny_img = cv2.Canny(blur_img, 100, 200)
        # 轮廓查找
        contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for region in contours:
            area = cv2.contourArea(region)   # 计算每个轮廓大小
            x, y, w, h = cv2.boundingRect(region)    # 近似轮廓矩形
            if area > 10000:    # 筛选轮廓，避免不需要的轮廓，具体还需要现场调试
                '''
                在原图上标上矩形面积
                '''
                cv2.putText(frame_copy1, f'Area: {area:.2f}', (x, y - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                '在原图上画出矩形'
                img_out = cv2.rectangle(frame_copy1, (x, y), (x + w, y + h), (0, 0, 255), 2)
                '''
                对裁剪的矩形进行预处理
                '''
                crop_img = frame_copy2[y:y + h, x:x + w,]
                cv2.imshow("crop_img", crop_img)
                wait_match_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                wait_match_img = cv2.threshold(wait_match_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                cv2.imshow("thr_img", wait_match_img)

                for index, each_template in enumerate(template):
                    index += 1
                    '''
                    模板匹配要求与原图一致类型调节类型和大小
                    '''
                    if wait_match_img.shape != each_template.shape or wait_match_img.dtype != each_template.dtype:
                        wait_match_img = cv2.resize(wait_match_img, (each_template.shape[1], each_template.shape[0]), interpolation=cv2.INTER_AREA)
                        wait_match_img = wait_match_img.astype(each_template.dtype)
                    '''
                    进行模板匹配
                    '''
                    result_img = cv2.matchTemplate(wait_match_img, each_template, cv2.TM_CCORR_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_img)
                    # 如果模板匹配
                    if max_val >= 0.5:  # 假设匹配阈值为0.5
                        cv2.rectangle(frame_copy3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.imshow("Matched Template", each_template)
                        # index 是匹配编号的索引，max_val 是准确率的数值
                        # 计算文本的宽度和高度
                        match_num = f'Match Num: {index}'
                        accuracy_text = f'Accuracy: {max_val * 100:.2f}%'
                        text_size = cv2.getTextSize(match_num, cv2.FONT_HERSHEY_COMPLEX, 1, 1)[0]
                        text_width = text_size[0]
                        # 计算下一行文本的起始位置
                        text_height = cv2.getTextSize(match_num, cv2.FONT_HERSHEY_COMPLEX, 1, 1)[1]
                        y_text2 = y - text_height - 20  # 两行文本之间有20像素的间距
                        # 放置第一行文本
                        cv2.putText(frame_copy3, match_num, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                        # 放置第二行文本
                        cv2.putText(frame_copy3, accuracy_text, (x, y_text2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                                    1)


        cv2.imshow("rectangle_frame", frame_copy1)
        cv2.imshow("Accuracy", frame_copy3)

    # 退出按键
    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        break