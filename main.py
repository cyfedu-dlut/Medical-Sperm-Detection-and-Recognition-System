import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
from sklearn.cluster import KMeans
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def is_close(bbox1, bbox2, threshold=50):
    """判断两个边界框是否过近"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    center1 = (x1 + w1 / 2, y1 + h1 / 2)
    center2 = (x2 + w2 / 2, y2 + h2 / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance < threshold

def merge_overlapping_boxes(boxes, overlap_thresh=0.6):
    """合并重叠的框"""
    if not boxes:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = boxes[:, 2] * boxes[:, 3]
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].tolist()

def is_background(bbox, img_shape, gray_img):
    """判断边界框是否属于背景或非目标物体（如尺度标尺）"""
    height, width = img_shape[:2]
    x, y, w, h = bbox

    # 计算边界框中心点
    center_x = x + w / 2
    center_y = y + h / 2

    # 判断是否在右下角区域内(可能是尺度标尺)
    in_bottom_right = (center_x > width * 0.85 and center_y > height * 0.85)

    # 获取区域的灰度图像
    roi = gray_img[y:y + h, x:x + w]
    if roi.size == 0:  # 处理边界情况
        return True

    # 计算区域的特征
    mean_intensity = np.mean(roi)
    std_intensity = np.std(roi)

    # 判断是否是背景
    # 1. 极亮或极暗的区域可能是背景
    is_extreme_intensity = mean_intensity < 30 or mean_intensity > 225
    # 2. 方差很小的区域可能是均匀背景
    is_uniform = std_intensity < 10
    # 3. 极端的宽高比可能是尺度标尺
    aspect_ratio = w / h if h > 0 else float('inf')
    is_strange_shape = aspect_ratio > 5.0 or aspect_ratio < 0.2

    # 综合判断
    return in_bottom_right or (is_extreme_intensity and is_uniform) or is_strange_shape

def extract_features(bbox, img):
    """提取边界框内目标的特征，包括紫色区域特征"""
    x, y, w, h = bbox

    # 确保输入图像是彩色的BGR格式
    if len(img.shape) == 2:  # 如果是灰度图
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    roi = img[y:y + h, x:x + w]

    if roi.size == 0:  # 处理边界情况
        return {
            "total_intensity": 0,
            "mean_intensity": 0,
            "std_intensity": 0,
            "area": 0,
            "aspect_ratio": 1.0,
            "perimeter_area_ratio": 0,
            "purple_intensity": 0,
            "purple_pixel_count": 0,
            "purple_ratio": 0
        }

        # 转换为HSV格式以便识别紫色区域
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 定义紫色范围 (可能需要根据实际紫色色调调整)
    # HSV中紫色大约在[120, 20, 50]到[170, 255, 255]范围
    lower_purple1 = np.array([130, 40, 80])
    upper_purple1 = np.array([160, 255, 230])

    # 添加第二个紫色区间，捕获更多变体
    lower_purple2 = np.array([145, 30, 100])
    upper_purple2 = np.array([165, 200, 255])

    # 创建紫色区域的掩码
    purple_mask1 = cv2.inRange(hsv_roi, lower_purple1, upper_purple1)
    purple_mask2 = cv2.inRange(hsv_roi, lower_purple2, upper_purple2)
    purple_mask = cv2.bitwise_or(purple_mask1, purple_mask2)

    # 提取紫色区域特征
    purple_pixel_count = np.sum(purple_mask > 0)
    purple_ratio = purple_pixel_count / (w * h) if (w * h) > 0 else 0

    # 如果有紫色像素，计算其强度
    if purple_pixel_count > 0:
        purple_region = cv2.bitwise_and(roi, roi, mask=purple_mask)
        purple_intensity = np.sum(purple_region) / purple_pixel_count
    else:
        purple_intensity = 0

    # 转为灰度图进行传统特征计算
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 计算基本特征
    total_intensity = np.sum(gray_roi)
    mean_intensity = np.mean(gray_roi)
    std_intensity = np.std(gray_roi)
    area = w * h
    aspect_ratio = w / h if h > 0 else 1.0

    # 计算轮廓特征
    _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(main_contour, True)
        perimeter_area_ratio = perimeter / np.sqrt(area) if area > 0 else 0
    else:
        perimeter_area_ratio = 0

    return {
        "total_intensity": total_intensity,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "perimeter_area_ratio": perimeter_area_ratio,
        "purple_intensity": purple_intensity,
        "purple_pixel_count": purple_pixel_count,
        "purple_ratio": purple_ratio
    }

def classify_objects(objects_with_features, num_classes=5):
    """使用K-means聚类对目标进行分类，优先考虑紫色区域特征，并确保按数量排序"""
    if len(objects_with_features) <= num_classes:
        # 如果对象不足num_classes个，每个对象一个类别，剩余类别为空
        class_dict = {f'类别{i + 1}': [] for i in range(num_classes)}
        for i, (bbox, features) in enumerate(objects_with_features):
            class_dict[f'类别{i + 1}'].append(bbox)
        return class_dict

        # 准备特征数据
    feature_matrix = []
    for _, features in objects_with_features:
        # 检查是否包含紫色相关特征
        has_purple_features = 'purple_intensity' in features and 'purple_pixel_count' in features

        if has_purple_features:
            # 如果有紫色特征，优先使用并增强其权重
            feature_vec = [
                features.get('purple_intensity', 0) * 2.0,  # 增强紫色强度权重
                features.get('purple_pixel_count', 0) * 1.5,  # 增强紫色像素数量权重
                features.get('purple_ratio', 0) * 3.0,  # 大幅增强紫色比例权重
                features['total_intensity'],
                features['mean_intensity'],
                features['std_intensity'],
                features['area']
            ]
        else:
            # 否则使用现有特征
            feature_vec = [
                features['total_intensity'],
                features['mean_intensity'],
                features['std_intensity'],
                features['area'],
                features['aspect_ratio'],
                features['perimeter_area_ratio']
            ]
        feature_matrix.append(feature_vec)

        # 标准化特征
    feature_matrix = np.array(feature_matrix)
    if feature_matrix.shape[0] > 0:  # 确保有数据
        # 对每一列进行标准化
        for i in range(feature_matrix.shape[1]):
            if np.std(feature_matrix[:, i]) > 0:  # 避免除以零
                feature_matrix[:, i] = (feature_matrix[:, i] - np.mean(feature_matrix[:, i])) / np.std(
                    feature_matrix[:, i])

    # 应用K-means聚类
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(feature_matrix)

    # 计算每个类别的对象数量
    class_counts = {}
    for i in range(num_classes):
        class_counts[i] = np.sum(clusters == i)

    # 按数量从大到小排序类别
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    # 创建新的类别映射：确保最大类别为类别1，第二大为类别2，依此类推
    class_mapping = {}
    for new_idx, (old_idx, _) in enumerate(sorted_classes):
        class_mapping[old_idx] = new_idx

    # 分析聚类中心，确定主要特征
    centers = kmeans.cluster_centers_

    # 根据特征维度判断排序标准（仅用于描述）
    if centers.shape[1] > 6:  # 有紫色特征
        class_description = "紫色区域"
    else:  # 没有紫色特征
        class_description = "像素强度"

    # 根据数量排序分配类别
    class_dict = {f'类别{i + 1}': [] for i in range(num_classes)}
    for i, (bbox, _) in enumerate(objects_with_features):
        cluster_id = clusters[i]
        class_id = class_mapping[cluster_id]
        class_dict[f'类别{class_id + 1}'].append(bbox)

        # 添加类别描述
    result_classes = {}
    result_classes[f'类别1 (最多对象，高{class_description})'] = class_dict['类别1']
    for i in range(2, num_classes):
        result_classes[f'类别{i}'] = class_dict[f'类别{i}']
    result_classes[f'类别5 (最少对象，低{class_description})'] = class_dict['类别5']

    return result_classes

def detect_tadpoles(image_path):
    """添加可视化的精子检测函数"""
    # 创建输出目录
    vis_dir = os.path.join(os.path.dirname(image_path), "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    # 创建可视化展示的图像列表
    visualization_images = []

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None, None, None

    # 保存和显示原始图像
    original = image.copy()
    visualization_images.append(("1. 原始图像", cv2.cvtColor(original, cv2.COLOR_BGR2RGB)))

    img_height, img_width = image.shape[:2]
    print(f"图像尺寸: {img_width} x {img_height}")

    # 转换到HSV色彩空间和灰度图
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 可视化HSV和灰度图
    hsv_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)  # 转换HSV到RGB用于显示
    visualization_images.append(("2. HSV色彩空间", hsv_display))
    visualization_images.append(("3. 灰度图", gray))

    print("正在处理图像，并逐步可视化...")
    # 创建3个紫色区间掩码，以覆盖所有紫色变体
    # 1. 深紫色（高饱和度，低亮度）
    lower_purple_dark = np.array([125, 100, 30])
    upper_purple_dark = np.array([170, 255, 120])
    purple_mask_dark = cv2.inRange(hsv, lower_purple_dark, upper_purple_dark)
    # 可视化紫色掩码——深紫色
    visualization_images.append(("4. 深紫色掩码", purple_mask_dark))
    # 在原图上叠加紫色掩码展示
    purple_overlay = original.copy()
    purple_overlay[purple_mask_dark > 0] = [255, 0, 255]  # 用紫色标记检测区域
    visualization_images.append(("5. 深紫色掩码叠加", cv2.cvtColor(purple_overlay, cv2.COLOR_BGR2RGB)))

    # 2. 中等紫色（中等饱和度和亮度）
    lower_purple_medium = np.array([125, 40, 100])
    upper_purple_medium = np.array([170, 150, 200])
    purple_mask_medium = cv2.inRange(hsv, lower_purple_medium, upper_purple_medium)
    # 可视化紫色掩码——中等紫色
    visualization_images.append(("4. 中等紫色掩码", purple_mask_medium))
    # 在原图上叠加中等紫色掩码展示
    purple_overlay = original.copy()
    purple_overlay[purple_mask_medium > 0] = [255, 0, 255]  # 用紫色标记检测区域
    visualization_images.append(("5. 中等紫色掩码叠加", cv2.cvtColor(purple_overlay, cv2.COLOR_BGR2RGB)))

    # 3. 浅紫色（低饱和度，高亮度 - 接近粉色和淡紫色）
    lower_purple_light = np.array([125, 15, 180])
    upper_purple_light = np.array([170, 80, 255])
    purple_mask_light = cv2.inRange(hsv, lower_purple_light, upper_purple_light)
    # 可视化紫色掩码——浅紫色
    visualization_images.append(("4. 浅紫色掩码", purple_mask_light))
    # 在原图上叠加浅紫色掩码展示
    purple_overlay = original.copy()
    purple_overlay[purple_mask_light > 0] = [255, 0, 255]  # 用紫色标记检测区域
    visualization_images.append(("5. 浅紫色掩码叠加", cv2.cvtColor(purple_overlay, cv2.COLOR_BGR2RGB)))

    # 合并所有紫色掩码
    purple_mask_full = cv2.bitwise_or(
        purple_mask_dark,
        cv2.bitwise_or(purple_mask_medium, purple_mask_light)
    )
    # 在原图上叠加紫色掩码2展示
    purple_overlay = original.copy()
    purple_overlay[purple_mask_full > 0] = [255, 0, 255]  # 用紫色标记检测区域
    visualization_images.append(("5. 所有紫色掩码叠加", cv2.cvtColor(purple_overlay, cv2.COLOR_BGR2RGB)))

    # 2. 提取深色区域
    _, dark_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # 可视化深色掩码
    visualization_images.append(("6. 深色掩码", dark_mask))

    # 在原图上叠加深色掩码展示
    dark_overlay = original.copy()
    dark_overlay[dark_mask > 0] = [0, 0, 255]  # 用红色标记深色区域
    visualization_images.append(("7. 深色掩码叠加", cv2.cvtColor(dark_overlay, cv2.COLOR_BGR2RGB)))

    # 3. 合并掩码和形态学处理
    combined_mask = cv2.bitwise_or(purple_mask_full,dark_mask)
    visualization_images.append(("8. 合并掩码(原始)", combined_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    visualization_images.append(("9. 开运算后掩码", opened_mask))

    combined_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    visualization_images.append(("10. 闭运算后掩码(最终)", combined_mask))

    # 在原图上叠加最终掩码
    final_mask_overlay = original.copy()
    final_mask_overlay[combined_mask > 0] = [0, 255, 0]  # 用绿色标记最终检测区域
    visualization_images.append(("11. 最终掩码叠加", cv2.cvtColor(final_mask_overlay, cv2.COLOR_BGR2RGB)))

    # 4. 找到轮廓,没有用合并的，而是发现dark_mask效果最好
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 可视化所有轮廓
    contour_image = original.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 255), 2)
    visualization_images.append(("12. 所有检测到的轮廓", cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)))

    # 5. 处理检测到的轮廓
    bboxes = []
    filtered_contour_image = original.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            # 过滤掉背景
            if not is_background((x, y, w, h), image.shape, gray):
                bboxes.append((x, y, w, h))
                cv2.rectangle(filtered_contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    visualization_images.append(("13. 初步筛选后的边界框", cv2.cvtColor(filtered_contour_image, cv2.COLOR_BGR2RGB)))

    # 6. 特殊尺寸目标的二次分割
    final_bboxes = []
    split_visualization = original.copy()

    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        area = w * h
        aspect_ratio = w / h if h > 0 else float('inf')

        # 判断是否需要分割的条件
        need_split = (area > 7000) and (aspect_ratio > 1.0 or aspect_ratio < 0.8)
        if need_split:
            cv2.rectangle(split_visualization, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色表示可能需要分割

            # 创建单个区域的可视化流程
            roi_vis_dir = os.path.join(vis_dir, f"roi_{i}")
            os.makedirs(roi_vis_dir, exist_ok=True)

            # 保存ROI区域
            roi = combined_mask[y:y + h, x:x + w].copy()
            cv2.imwrite(os.path.join(roi_vis_dir, "1_roi_mask.png"), roi)

            roi_original = original[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(roi_vis_dir, "2_roi_original.png"), roi_original)

            if roi.size == 0:  # 处理边界情况
                continue

            # 距离变换
            dist_transform = cv2.distanceTransform(roi, cv2.DIST_L2, 5)

            # 归一化距离变换用于可视化
            dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(os.path.join(roi_vis_dir, "3_distance_transform.png"), dist_norm)

            # 生成热图可视化
            plt.figure(figsize=(8, 6))
            plt.imshow(dist_transform, cmap='jet')
            plt.colorbar(label='Distance')
            plt.title('Distance Transform Heatmap')
            plt.savefig(os.path.join(roi_vis_dir, "4_distance_heatmap.png"))
            plt.close()

            # 计算自适应阈值
            complexity = np.std(dist_transform) / np.mean(dist_transform) if np.mean(dist_transform) > 0 else 1
            threshold_factor = max(0.05, min(0.5, 0.15 + 0.1 * complexity))
            threshold_value = threshold_factor * dist_transform.max()

            # 保存阈值信息
            with open(os.path.join(roi_vis_dir, "5_threshold_info.txt"), "w") as f:
                f.write(f"Complexity: {complexity:.4f}\n")
                f.write(f"Threshold Factor: {threshold_factor:.4f}\n")
                f.write(f"Max Distance: {dist_transform.max():.4f}\n")
                f.write(f"Final Threshold: {threshold_value:.4f}\n")

            # 阈值处理
            _, sure_fg = cv2.threshold(dist_transform, threshold_value, 255, 0)
            sure_fg = np.uint8(sure_fg)
            # cv2.imwrite(os.path.join(roi_vis_dir, "6_thresholded.png"), sure_fg)

            # 可视化阈值效果
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.imshow(roi, cmap='gray')
            plt.title('Original ROI')

            plt.subplot(2, 2, 2)
            plt.imshow(dist_transform, cmap='jet')
            plt.title('Distance Transform')
            plt.colorbar()

            plt.subplot(2, 2, 3)
            # 检查dist_transform是否有效并且不全为零或常数
            if np.any(dist_transform) and np.std(dist_transform) > 0:
                try:
                    plt.hist(dist_transform.ravel(), bins=min(50, len(np.unique(dist_transform))))
                    plt.axvline(threshold_value, color='r', linestyle='dashed')
                    plt.title(f'Histogram with Threshold={threshold_value:.2f}')
                except Exception as e:
                    # 出错时显示简单的替代图
                    plt.text(0.5, 0.5, f"Cannot create histogram: {str(e)[:40]}...",
                             ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title("Histogram Error")
            else:
                # 处理距离变换全零或常数的情况
                plt.text(0.5, 0.5, "Cannot create histogram: uniform distance data",
                         ha='center', va='center', transform=plt.gca().transAxes)
                plt.title("No Histogram Available")

            plt.subplot(2, 2, 4)
            plt.imshow(sure_fg, cmap='gray')
            plt.title('Thresholded Result')

            plt.tight_layout()
            plt.savefig(os.path.join(roi_vis_dir, "7_threshold_comparison.png"))
            plt.close()

            # 连通组件分析
            num_labels, labels = cv2.connectedComponents(sure_fg)

            # 可视化标签
            label_heatmap = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
            colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
            colors[0] = [0, 0, 0]  # 背景为黑色

            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    label_heatmap[i, j] = colors[labels[i, j]]

            cv2.imwrite(os.path.join(roi_vis_dir, f"8_connected_components_{num_labels - 1}_regions.png"),label_heatmap)

            # 可视化连通组件结果
            plt.figure(figsize=(8, 6))
            plt.imshow(label_heatmap)
            plt.title(f'Connected Components ({num_labels - 1} regions)')
            plt.savefig(os.path.join(roi_vis_dir, "9_connected_components_colored.png"))
            plt.close()

            # 判断是否是单一连通区域（不计背景）- 这是核心修改
            if num_labels <= 2:  # 只有背景(0)和一个前景(1)
                # 记录误判情况
                with open(os.path.join(roi_vis_dir, "10_decision.txt"), "w") as f:
                    f.write("Single connected region detected: False positive splitting candidate.\n")
                    f.write("Decision: Keep original ROI without splitting.\n")

                # 添加原始边界框到最终结果列表
                final_bboxes.append(bbox)

                # 在可视化图像上用黄色标记这个区域
                cv2.rectangle(split_visualization, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 黄色表示保留原始框
            else:
                # 有多个连通区域，需要分割处理
                with open(os.path.join(roi_vis_dir, "10_decision.txt"), "w") as f:
                    f.write(f"Detected {num_labels - 1} connected components.\n")
                    f.write("Decision: Proceed with splitting.\n")

                # 处理每个连通区域
                sub_regions_img = original[y:y + h, x:x + w].copy()
                found_valid_regions = False

                for label in range(1, num_labels):
                    mask = np.uint8(labels == label)
                    label_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # 保存单个标签的掩码
                    cv2.imwrite(os.path.join(roi_vis_dir, f"11_label_{label}_mask.png"), mask * 255)

                    for cnt in label_contours:
                        sub_area = cv2.contourArea(cnt)
                        if sub_area > 250:  # 过滤太小的区域
                            sub_x, sub_y, sub_w, sub_h = cv2.boundingRect(cnt)
                            if sub_w > 3 and sub_h > 3:
                                found_valid_regions = True

                                # 在子区域图像上绘制边界框
                                cv2.rectangle(sub_regions_img, (sub_x, sub_y),
                                              (sub_x + sub_w, sub_y + sub_h),
                                              colors[label].tolist(), 2)

                                # 转换回原图坐标
                                new_bbox = (x + sub_x, y + sub_y, sub_w, sub_h)

                                # 再次检查不是背景
                                if not is_background(new_bbox, image.shape, gray):
                                    final_bboxes.append(new_bbox)

                                    # 在最终结果上绘制分割后的边界框
                                    cv2.rectangle(split_visualization,
                                                  (new_bbox[0], new_bbox[1]),
                                                  (new_bbox[0] + new_bbox[2], new_bbox[1] + new_bbox[3]),
                                                  (0, 255, 0), 2)  # 绿色表示分割后的框

                # 保存子区域结果
                cv2.imwrite(os.path.join(roi_vis_dir, "12_sub_regions_result.png"), sub_regions_img)

                # 如果未找到任何有效子区域，回退到原始边界框
                if not found_valid_regions:
                    final_bboxes.append(bbox)
                    cv2.rectangle(split_visualization, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 黄色表示保留的原框

                    with open(os.path.join(roi_vis_dir, "13_fallback.txt"), "w") as f:
                        f.write("No valid sub-regions found after filtering.\n")
                        f.write("Decision: Fallback to original bounding box.\n")
        else:
            # 不需要分割的区域
            final_bboxes.append(bbox)
            cv2.rectangle(split_visualization, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 黄色表示不需要分割的框
    # 保存分割可视化结果
    visualization_images.append(("14. 分割前后的边界框", cv2.cvtColor(split_visualization, cv2.COLOR_BGR2RGB)))

    # 合并重叠的框
    pre_merge_vis = original.copy()
    for bbox in final_bboxes:
        x, y, w, h = bbox
        cv2.rectangle(pre_merge_vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

    visualization_images.append(("15. 合并前的边界框", cv2.cvtColor(pre_merge_vis, cv2.COLOR_BGR2RGB)))

    final_bboxes = merge_overlapping_boxes(final_bboxes)

    post_merge_vis = original.copy()
    for bbox in final_bboxes:
        x, y, w, h = bbox
        cv2.rectangle(post_merge_vis, (x, y), (x + w, y + h), (255, 0, 127), 2)

    visualization_images.append(("16. 合并后的最终边界框", cv2.cvtColor(post_merge_vis, cv2.COLOR_BGR2RGB)))

    # 7. 提取特征并进行分类
    objects_with_features = []
    for bbox in final_bboxes:
        features = extract_features(bbox, gray)
        objects_with_features.append((bbox, features))

    # 可视化特征数据
    feature_data = {
        'X坐标': [bbox[0] for bbox, _ in objects_with_features],
        'Y坐标': [bbox[1] for bbox, _ in objects_with_features],
        '宽度': [bbox[2] for bbox, _ in objects_with_features],
        '高度': [bbox[3] for bbox, _ in objects_with_features],
        '面积': [feat['area'] for _, feat in objects_with_features],
        '宽高比': [feat['aspect_ratio'] for _, feat in objects_with_features],
        '平均强度': [feat['mean_intensity'] for _, feat in objects_with_features],
        '总强度': [feat['total_intensity'] for _, feat in objects_with_features],
        '标准差': [feat['std_intensity'] for _, feat in objects_with_features]
    }

    # 创建特征散点图矩阵
    plt.figure(figsize=(16, 12))
    feature_keys = ['平均强度', '总强度', '面积', '宽高比']
    n = len(feature_keys)

    for i in range(n):
        for j in range(n):
            plt.subplot(n, n, i * n + j + 1)
            if i == j:
                # 对角线上绘制直方图
                plt.hist(feature_data[feature_keys[i]], bins=20)
                plt.title(feature_keys[i])
            else:
                # 非对角线绘制散点图
                plt.scatter(feature_data[feature_keys[j]], feature_data[feature_keys[i]], alpha=0.6)
                plt.xlabel(feature_keys[j])
                plt.ylabel(feature_keys[i])

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "feature_scatter_matrix.png"))
    plt.close()

    # 根据特征聚类分类
    color_classes = classify_objects(objects_with_features)

    # 可视化聚类结果
    cluster_vis = original.copy()
    colors = {
        '类别1 (最多对象，高紫色区域)': (0, 0, 255),  # 红色
        '类别2': (255, 0, 0),  # 蓝色
        '类别3': (0, 255, 0),  # 绿色
        '类别4': (255, 255, 0),  # 青色
        '类别5 (最少对象，低紫色区域)': (255, 0, 255)  # 紫色
    }

    for class_name, boxes in color_classes.items():
        color = colors[class_name]
        for bbox in boxes:
            x, y, w, h = bbox
            cv2.rectangle(cluster_vis, (x, y), (x + w, y + h), color, 4)

    visualization_images.append(("17. 聚类分类结果", cv2.cvtColor(cluster_vis, cv2.COLOR_BGR2RGB)))

    # 创建大型可视化图
    rows = 5
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    fig.suptitle("精子检测可视化流程", fontsize=16)

    for i, (title, img) in enumerate(visualization_images):
        if i < rows * cols:
            row = i // cols
            col = i % cols
            axes[row, col].imshow(img)
            axes[row, col].set_title(title)
            axes[row, col].axis('off')

    # 隐藏空白子图
    for i in range(len(visualization_images), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(vis_dir, "complete_visualization.png"), dpi=300)
    plt.close()

    # 保存每一个单独的可视化步骤
    for i, (title, img) in enumerate(visualization_images):
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{i + 1:02d}_{title.replace(' ', '_').replace(':', '_')}.png"), dpi=200)
        plt.close()

    box_sums = [(bbox, features['total_intensity']) for bbox, features in objects_with_features]

    # 绘制最终结果
    result = original.copy()

    # 绘制边界框
    for class_name, boxes in color_classes.items():
        color = colors[class_name]
        for bbox in boxes:
            x, y, w, h = bbox
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 4)

            # 创建带有图例的完整显示图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8),
                                   gridspec_kw={'width_ratios': [4, 1]})

    # 显示主图像
    ax1.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    ax1.set_title('医学精子检测结果', fontsize=12)

    # 创建图例
    legend_elements = []
    class_counts = []
    for class_name, boxes in color_classes.items():
        count = len(boxes)
        legend_elements.append(f'{class_name}: {count}个')
        class_counts.append(count)

        # 在右侧子图中创建柱状图作为图例
    y_pos = np.arange(len(legend_elements))
    bars = ax2.barh(y_pos, class_counts,
                    color=[tuple(np.array(list(colors.values())[i])[::-1] / 255)
                           for i in range(len(colors))])

    # 在柱状图上添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{int(width)}个',
                 ha='left', va='center', fontsize=10)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(legend_elements)
    ax2.set_title('检测统计', fontsize=12)
    ax2.invert_yaxis()

    # 调整布局
    plt.tight_layout()

    # 保存结果
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_result.jpg'
    output_path = os.path.join(os.path.dirname(image_path), output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    print(f"结果已保存到 {output_path}")

    # 保存可视化目录信息
    print(f"详细可视化步骤已保存到 {vis_dir}")

    # 显示结果
    plt.show()

    return result, color_classes, box_sums

def main():
    import os

    # 获取当前目录下的所有jpg文件
    image_dir = './spermimage'
    if not os.path.exists(image_dir):
        print(f"警告：目录 {image_dir} 不存在，使用当前目录")
        image_dir = '.'

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"在 {image_dir} 中未找到图像文件")
        return

    print(f"找到以下图像文件：")
    for i, file in enumerate(image_files):
        print(f"{i + 1}. {file}")

    # 可以选择处理单个文件还是处理所有文件
    choice = input("输入文件编号处理单个文件，输入'all'处理所有文件：")

    if choice.lower() == 'all':
        # 处理所有文件
        for file in image_files:
            image_path = os.path.join(image_dir, file)
            print(f"\n正在处理：{image_path}")
            result, statistics, box_sums = detect_tadpoles(image_path)

            if result is not None:
                # 打印统计信息
                print("\n检测统计结果:")
                for class_name, boxes in statistics.items():
                    print(f"{class_name}: {len(boxes)}个目标")
    else:
        try:
            # 处理单个文件
            file_idx = int(choice) - 1
            if 0 <= file_idx < len(image_files):
                image_path = os.path.join(image_dir, image_files[file_idx])
                print(f"\n正在处理：{image_path}")
                result, statistics, box_sums = detect_tadpoles(image_path)

                if result is not None:
                    # 打印统计信息
                    print("\n检测统计结果:")
                    for class_name, boxes in statistics.items():
                        print(f"{class_name}: {len(boxes)}个目标")
            else:
                print("无效的文件编号")
        except ValueError:
            print("无效的输入，请输入数字或'all'")


if __name__ == "__main__":
    main()