import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_detection_steps(image, hsv, purple_mask, dark_mask,
                               combined_mask, open_mask, close_mask):
    plt.figure(figsize=(20, 12))

    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(hsv)
    plt.title('HSV色彩空间')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(purple_mask)
    plt.title('紫色区域掩码')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.title('灰度图')
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.imshow(dark_mask)
    plt.title('深色区域掩码')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(combined_mask)
    plt.title('合并掩码')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(open_mask)
    plt.title('开运算后掩码')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(close_mask)
    plt.title('闭运算后掩码')
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle('图像处理关键步骤可视化', fontsize=16)
    plt.subplots_adjust(top=0.9)

    visualization_dir = 'visualization'
    os.makedirs(visualization_dir, exist_ok=True)
    plt.savefig(f'{visualization_dir}/{len(os.listdir(visualization_dir))+1:03d}_image_processing_steps.png')
    plt.close()