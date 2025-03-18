# 🔬 精子智能图像检测与分类系统 V2.0 

完整代码运行：code.ipynb

<img src="./demoimage/complete_visualization.png">
<img src="./demoimage/feature_scatter_matrix.png">  

## 📝 项目简介  

本项目是一个基于计算机视觉和图像处理技术的生物医学图像分析系统，专注于精子的智能检测、分类和可视化。通过先进的图像处理算法，项目能够从复杂的生物医学图像中准确识别、定位和分类目标。
<img src="./demoimage/01_result.jpg">
<img src="./demoimage/02_result.jpg">
<img src="./demoimage/03_result.jpg">
<img src="./demoimage/04_result.jpg">
<img src="./demoimage/05_result.jpg">

### 🌟 主要特点  

- 🖼️ 多步骤图像处理流程  
- 🔍 精确的检测算法  
- 📊 多维度目标分类
- 📈 详细的可视化结果展示 
- 🧩 模块化的代码架构  

## 🚀 快速开始  
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

### 前置条件  

- Python 3.8+  
- pip 包管理器  

### 安装步骤  

1. 克隆项目仓库  
```bash  
git clone https://github.com/yourusername/tadpole-detection.git  
cd tadpole-detection  
创建虚拟环境（推荐）
python -m venv venv  
source venv/bin/activate  # 在 Windows 上使用 `venv\Scripts\activate`  
安装依赖
pip install -r requirements.txt  
📖 使用指南
基本运行
python main.py    

📌 注意事项
确保输入图像清晰、对比度适中
建议使用 JPG 或 PNG 格式
大型或极其复杂的图像可能需要调整算法参数
🔒 许可证
本项目基于 MIT 许可证 - 详见 LICENSE 文件

免责声明：本项目仅用于学术研究和教育目的，不应直接用于临床诊断。

🌐 联系方式
项目主页: [https://github.com/cyfedu-dlut/Medical-Sperm-Detection-and-Recognition-System]
邮箱: yfcao@mail.dlut.edu.cn
个人博客/主页: [https://cyfedu-dlut.github.io/PersonalWeb/]