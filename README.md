 
# 🔬 Intelligent Sperm Image Detection and Classification System V2.0 
click 【[中文版本](./README_CH.md)】
complete running: code.ipynb

<img src="./demoimage/complete_visualization.png">
<img src="./demoimage/feature_scatter_matrix.png">  

## 📝 Project Overview  

This project is a biomedical image analysis system based on computer vision and image processing techniques, focusing on intelligent detection, classification, and visualization of sperm. Using advanced image processing algorithms, the project can accurately identify, locate, and classify targets from complex biomedical images.  
<img src="./demoimage/01_result.jpg">
<img src="./demoimage/02_result.jpg">
<img src="./demoimage/03_result.jpg">
<img src="./demoimage/04_result.jpg">
<img src="./demoimage/05_result.jpg">

### 🌟 Key Features  

- 🖼️ Multi-step Image Processing Workflow  
- 🔍 Precise Target Detection Algorithms  
- 📊 Multi-dimensional Target Classification  
- 📈 Detailed Visualization of Results  
- 🧩 Modular Code Architecture  

## 🚀 Quick Start  
<img src="./demoimage/01_1._原始图像.png">  
<img src="./demoimage/02_2._HSV色彩空间.png">  
<img src="./demoimage/03_3._灰度图.png">  
<img src="./demoimage/04_4._深紫色掩码.png">  
<img src="./demoimage/05_5._深紫色掩码叠加.png">  
<img src="./demoimage/06_4._中等紫色掩码.png">  
<img src="./demoimage/07_5._中等紫色掩码叠加.png">  
<img src="./demoimage/09_5._浅紫色掩码叠加.png">
<img src="./demoimage/10_5._所有紫色掩码叠加.png">  
<img src="./demoimage/11_6._深色掩码.png">  
<img src="./demoimage/12_7._深色掩码叠加.png">  
<img src="./demoimage/13_8._合并掩码(原始).png">  
<img src="./demoimage/14_9._开运算后掩码.png">
<img src="./demoimage/15_10._闭运算后掩码(最终).png">  
<img src="./demoimage/16_11._最终掩码叠加.png">  
<img src="./demoimage/17_12._所有检测到的轮廓.png">  
<img src="./demoimage/18_13._初步筛选后的边界框.png">  
<img src="./demoimage/19_14._分割前后的边界框.png">
<img src="./demoimage/20_15._合并前的边界框.png">  
<img src="./demoimage/21_16._合并后的最终边界框.png">  
<img src="./demoimage/22_17._聚类分类结果.png">  


### Prerequisites  

- Python 3.8+  
- pip Package Manager  

### Installation Steps  

1. Clone the Project Repository  
~~~bash  
git clone https://github.com/yourusername/tadpole-detection.git  
cd tadpole-detection
~~~

2. Create Virtual Environment (Recommended)
~~~bash
python -m venv venv  
source venv/bin/activate  # Use `venv\Scripts\activate` on Windows  
~~~

3. Install Dependencies
~~~bash
pip install -r requirements.txt  
~~~

# 📖 User Guide
Basic Execution
python main.py  
 
### 📌 Precautions
Ensure input images are clear with appropriate contrast
Recommended to use JPG or PNG formats
Large or extremely complex images may require algorithm parameter adjustments
### 🔒 License
This project is licensed under the MIT License - see the LICENSE file for details

### 🙌 Acknowledgments
OpenCV Development Team
NumPy Community
Matplotlib Project

Disclaimer: This project is for academic research and educational purposes only and should not be directly used for clinical diagnosis.

🌐 Contact
Project Homepage: [https://github.com/cyfedu-dlut/Medical-Sperm-Detection-and-Recognition-System]
Email: yfcao@mail.dlut.edu.cn
Personal Blog/Homepage: [https://cyfedu-dlut.github.io/PersonalWeb/]