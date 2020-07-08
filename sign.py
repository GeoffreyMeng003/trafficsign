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
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([124, 255, 255])
        # mask
        mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blurred = cv2.blur(mask, (9, 9))
        cv2.imshow('blurred', blurred)

        # binarization
        ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
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
        for con in contours:

            rect = cv2.minAreaRect(con)
            # box
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # the segmation area on original image
            cv2.drawContours(res, [box], -1, (0, 0, 255), 2)
            print([box])
            # the dimension of matrix
            h1 = min(box.max(axis=0))
            h2 = min(box.min(axis=0))
            l1 = max(box.max(axis=1))
            l2 = min(box.max(axis=1))
            print('h1', h1)
            print('h2', h2)
            print('l1', l1)
            print('l2', l2)
            # make sure if the area is accurate
            if h1 - h2 > 0 and l1 - l2 > 0:
                # segmentation
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
