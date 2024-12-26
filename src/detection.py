import cv2
import numpy as np
import os

def is_abnormally_large_box(bbox, bboxes, width_threshold=3, height_threshold=3):
    """检查当前框是否异常大"""
    x, y, w, h = bbox
    current_area = w * h  # 当前框面积

    for other_bbox in bboxes:
        if other_bbox == bbox:
            continue
        ox, oy, ow, oh = other_bbox
        other_area = ow * oh  # 其他框面积

        width_ratio_1 = ow / w
        height_ratio_1 = oh / h

        if (width_ratio_1 > width_threshold or height_ratio_1 > height_threshold):
            print(f"异常检测框：当前框 {(w,h)}，比较框 {(ow,oh)}")
            return True
    return False

def extract_small_bboxes(image, bbox, min_area=100, max_area=2000):
    """从异常大框中提取矩形框"""
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray_roi, -1, kernel)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for cnt in contours:
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        area = w1 * h1
        if min_area < area < max_area:
            rectangles.append((x + x1, y + y1, w1, h1))

    return rectangles