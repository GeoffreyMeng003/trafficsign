import os
import numpy as np
import cv2




def cutVideos2Pictures(video_path='Training.avi', saveDir='pictures/myself/'):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    vc = cv2.VideoCapture(video_path)
    c = 1
    if vc.isOpened():
        cap, frame = vc.read()
    else:
        cap = False
    while cap:
        cap, frame = vc.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_lower = np.array([100, 43, 43])
        blue_upper = np.array([124, 255, 255])
        # mask
        mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blurred=cv2.GaussianBlur(mask, (5, 5), 0)
        cv2.imshow('blurred', blurred)
        # binarization
        ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        #binary = cv2.Canny(blurred, 50, 150)
        cv2.imshow('blurred binary', binary)

        # closed
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # erode and dilate
        erode = cv2.erode(closed, None, iterations=4)
        dilate = cv2.dilate(erode, None, iterations=4)

        contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print('the number of contours：', len(contours))
        i = 0
        res = frame.copy()
        contours = sorted(contours, key=cv2.contourArea)
        for con in contours:
            # 轮廓转换为矩形
            rect = cv2.minAreaRect(con)
            # 矩形转换为box
            box = np.int0(cv2.boxPoints(rect))
            x, y, w, h = cv2.boundingRect(con)
            # 在原图画出目标区域
            print([box])
            # 计算矩形的行列
            h1 = max([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
            h2 = min([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
            l1 = max([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
            l2 = min([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
            print('h1', h1)
            print('h2', h2)
            print('l1', l1)
            print('l2', l2)
                # make sure if the area is accurate
            if h1-h2 > 0 and l1-l2 > 0 and cv2.contourArea(con) > 100 and (0.9 < w/h < 1.1):
                # segmentation
                cv2.drawContours(res, [box], -1, (0, 0, 255), 2)
                temp = frame[h2:h1, l2:l1]
                i = i + 1
                # show the image sign
                #cv2.imshow('sign' + str(i), temp)
                # turn it into 40*40
                #atemp = cv2.resize(temp, (40, 40), interpolation=cv2.INTER_CUBIC)
                # cv2.imshow('aftersign' + str(i), atemp)
                cv2.imwrite(saveDir + str(i) + str(c) + '.jpg', temp)
                cv2.imshow('res', res)
                c = c + 1
        cv2.waitKey(1)
    vc.release()


if __name__ == '__main__':
    cutVideos2Pictures(video_path='\Training.avi', saveDir='pictures/Training/')
