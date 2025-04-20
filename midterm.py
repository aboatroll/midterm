#!/usr/bin/env python
# coding: utf-8

# In[35]:


import cv2
import numpy as np

cap = cv2.VideoCapture("LaneVideo.mp4")

ww, hh, rh, r = 640, 400, 0.6, 3
xx1, yy1, xx2, yy2 = int(ww*0.4), int(hh*rh), int(ww*0.6), int(hh*rh)
p1, p2, p3, p4 = [r, hh-r], [ww-r, hh-r], [xx2, yy2], [xx1, yy2]

right_confirmed_points = None
left_confirmed_points = None
frame_count = 0
RESET_INTERVAL = 30  # 每 1 秒重置

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    frame_count += 1
    img = cv2.resize(img, (ww, hh))
    output = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    output = cv2.dilate(output, kernel)
    output = cv2.GaussianBlur(output, (5, 5), 0)
    output = cv2.erode(output, kernel)
    output = cv2.Canny(output, 150, 200)

    zero = np.zeros((hh, ww, 1), dtype='uint8')
    area = [p1, p2, p3, p4]
    pts = np.array(area)
    zone = cv2.fillPoly(zero, [pts], 255)
    output = cv2.bitwise_and(output, zone)

    lines = cv2.HoughLinesP(output, 1, np.pi / 180, 40, None, 15, 70)
    s1, s2, b1, b2 = 0, 0, 0, 0
    img2 = img.copy()
    new_left_confirmed = None
    new_right_confirmed = None

    if lines is not None:
        for i in range(len(lines)):
            l = lines[i][0]
            x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
            if x2 - x1 == 0:
                continue
            s = (y2 - y1) / (x2 - x1)
            b = y1 - s * x1
            if min(x1, x2) < 30 or max(x1, x2) > ww - 30:
                continue

            if s < 0 and s < s1:
                s1, b1 = s, b
                new_left_confirmed = (s1, b1)
            elif s > 0 and s > s2:
                s2, b2 = s, b
                new_right_confirmed = (s2, b2)

    # 每秒強制重新偵測（清除線條資訊）
    if frame_count % RESET_INTERVAL == 0:
        left_confirmed_points = None
        right_confirmed_points = None

    if new_left_confirmed:
        left_confirmed_points = new_left_confirmed
    if new_right_confirmed:
        right_confirmed_points = new_right_confirmed

    y1 = hh - r
    y2 = int(hh - hh * 0.175)

    if left_confirmed_points is not None:
        s1, b1 = left_confirmed_points
        x1_l = int((y1 - b1) / s1)
        x2_l = int((y2 - b1) / s1)
        cv2.line(img2, (x1_l, y1), (x2_l, y2), (0, 255, 0), 2)

    if right_confirmed_points is not None:
        s2, b2 = right_confirmed_points
        x1_r = int((y1 - b2) / s2)
        x2_r = int((y2 - b2) / s2)
        cv2.line(img2, (x1_r, y1), (x2_r, y2), (255, 255, 0), 2)

    cv2.imshow("Lane Detection", img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

