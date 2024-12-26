import os
import cv2
import matplotlib.pyplot as plt
from src.detection import extract_small_bboxes, is_abnormally_large_box
from src.utils import pixel_ratio_classification
from src.visualization import visualize_detection_steps


def detect_tadpoles(input_image):
    """
    主要的蝌蚪/精子检测流程函数

    参数:
        input_image (str): 输入图像路径

    返回:
        tuple: 处理结果、统计信息、边界框像素统计
    """
    # 读取图像
    image = cv2.imread(input_image)
    if image is None:
        raise ValueError(f"无法读取图像：{input_image}")

    original = image.copy()

    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 紫色区域的HSV范围
    lower_purple = (120, 40, 50)
    upper_purple = (210, 255, 255)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # 转换为灰度图用于深色区域检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 合并掩码
    combined_mask = cv2.bitwise_or(purple_mask, dark_mask)

    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    open_combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    close_combined_mask = cv2.morphologyEx(open_combined_mask, cv2.MORPH_CLOSE, kernel)

    # 可视化处理步骤
    visualize_detection_steps(
        image, hsv, purple_mask, dark_mask,
        combined_mask, open_combined_mask, close_combined_mask
    )

    # 寻找轮廓
    contours, _ = cv2.findContours(
        close_combined_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 初始化检测结果
    detected_bboxes = []

    # 处理每个轮廓
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 过滤小目标和异常大目标
        if w * h < 100 or w * h > 10000:
            continue

        bbox = (x, y, w, h)
        detected_bboxes.append(bbox)

        # 标记边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 处理异常大的边界框
    final_bboxes = []
    for bbox in detected_bboxes:
        if is_abnormally_large_box(bbox, detected_bboxes):
            # 对异常大框进行分割
            small_bboxes = extract_small_bboxes(original, bbox)
            final_bboxes.extend(small_bboxes)
        else:
            final_bboxes.append(bbox)

            # 像素比例分类
    color_classes, bbox_pixel_stats = pixel_ratio_classification(hsv, final_bboxes)

    # 绘制最终结果
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('目标检测结果')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('result_detection.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    return image, color_classes, bbox_pixel_stats


def print_detection_results(color_classes, bbox_pixel_stats):
    """
    打印检测结果

    参数:
        color_classes (dict): 分类结果
        bbox_pixel_stats (list): 边界框像素统计
    """
    print("\n检测统计结果:")
    for class_name, boxes in color_classes.items():
        print(f"{class_name}: {len(boxes)}个目标")

    print("\n各检测框像素比例:")
    for stat in bbox_pixel_stats:
        print(f"框 {stat['bbox']}: 浅紫色/深紫色比例 {stat['light_to_deep_purple_ratio']:.4f}")


def main():
    """主程序入口"""
    # 输入图像路径
    input_image = 'data/sperm.jpg'  # 确保图像存在

    try:
        # 执行检测
        result_image, color_classes, bbox_pixel_stats = detect_tadpoles(input_image)

        # 打印结果
        print_detection_results(color_classes, bbox_pixel_stats)

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()