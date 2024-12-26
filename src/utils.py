import cv2
import numpy as np

def pixel_ratio_classification(hsvimage, bboxes):
    """对bbox进行像素比例分类的后处理函数"""
    lower_deep_purple = np.array([150, 50, 50])
    upper_deep_purple = np.array([210, 255, 255])
    lower_light_purple = np.array([120, 40, 60])
    upper_light_purple = np.array([140, 255, 255])

    deep_purple_mask = cv2.inRange(hsvimage, lower_deep_purple, upper_deep_purple)
    light_purple_mask = cv2.inRange(hsvimage, lower_light_purple, upper_light_purple)

    bbox_pixel_stats = []

    for bbox in bboxes:
        x, y, w, h = bbox
        roi_deep_purple = deep_purple_mask[
                          max(0, y):min(hsvimage.shape[0], y + h),
                          max(0, x):min(hsvimage.shape[1], x + w)
                          ]
        roi_light_purple = light_purple_mask[
                           max(0, y):min(hsvimage.shape[0], y + h),
                           max(0, x):min(hsvimage.shape[1], x + w)
                           ]

        deep_purple_count = cv2.countNonZero(roi_deep_purple)
        light_purple_count = cv2.countNonZero(roi_light_purple)

        light_to_deep_purple_ratio = light_purple_count / deep_purple_count if deep_purple_count > 0 else 0.00001

        bbox_pixel_stats.append({
            'bbox': bbox,
            'deep_purple_count': deep_purple_count,
            'light_purple_count': light_purple_count,
            'light_to_deep_purple_ratio': light_to_deep_purple_ratio
        })

    bbox_pixel_stats.sort(key=lambda x: x['light_to_deep_purple_ratio'], reverse=True)

    color_classes = {
        '类别一': [], '类别二': [],
        '类别三': [], '类别四': [],
        '类别五': []
    }

    for stat in bbox_pixel_stats:
        ratio = stat['light_to_deep_purple_ratio']
        if ratio > 0.4:
            color_classes['类别一'].append(stat['bbox'])
        elif 0.3 <= ratio <= 0.4:
            color_classes['类别二'].append(stat['bbox'])
        elif 0.1 < ratio < 0.3:
            color_classes['类别三'].append(stat['bbox'])
        elif 0.01 < ratio <= 0.1:
            color_classes['类别四'].append(stat['bbox'])
        else:
            color_classes['类别五'].append(stat['bbox'])

    return color_classes, bbox_pixel_stats