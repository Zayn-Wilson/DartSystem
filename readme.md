# 飞镖系统 (DartSystem)


## 项目概述

飞镖系统是一个基于计算机视觉的应用程序，用于检测和跟踪绿色光点（如激光指示器或LED灯）。系统通过摄像头捕获图像，识别绿色光点的位置，并计算其相对于屏幕中心的偏移量，可用于飞镖游戏、交互式演示或其他需要光点跟踪的应用场景。

## 文件结构
DartSystem/
├── .idea/ # IDE配置文件
├── .venv/ # Python虚拟环境
├── opencv_green_detection.py # 主要的绿色光点检测和跟踪模块
├── weight_matrix_visualizer.py # 权重矩阵可视化工具
├── screen_recorder.py # 屏幕录制工具
├── main.py # 主程序文件
├── heatmap.png # 权重矩阵热力图
├── weight_matrix_viz.html # 权重矩阵可视化HTML页面
├── video/ # 视频保存文件夹
├── serial_communication.py # 串口通信模块
└── readme.md # 项目说明文档
111
## 主要功能

1. **绿色光点检测**：使用HSV颜色空间和形态学操作精确识别绿色光点
2. **位置跟踪**：计算光点相对于屏幕中心的水平偏移量
3. **稳定性增强**：
   - 卡尔曼滤波器用于平滑跟踪结果
   - 历史记录和中值滤波减少抖动
   - 目标丢失处理机制（持续返回最后有效值直到目标重新出现）
4. **权重矩阵**：通过权重矩阵增强中间区域检测，减少顶部和底部区域的干扰
5. **可视化工具**：提供权重矩阵的HTML可视化界面
6. **屏幕录制**：自动录制屏幕，便于后期分析和问题诊断

## 技术细节

### 绿色光点检测

系统使用以下步骤检测绿色光点：

1. 将RGB图像转换为HSV颜色空间
2. 应用高斯模糊减少噪声
3. 使用HSV阈值创建二值掩膜
4. 应用形态学操作（开运算和闭运算）去除噪点
5. 查找轮廓并筛选有效目标
6. 计算轮廓的中心点和面积

### 权重矩阵

权重矩阵用于增强特定区域的检测效果：

- 中间区域（图像高度的35%-65%）保持最高权重（1.0）
- 顶部和底部区域权重降低（0.2）
- 这种设计可以减少顶部和底部区域的干扰，如天花板灯光或地面反光

### 稳定性增强

系统使用多种方法提高跟踪稳定性：

1. **卡尔曼滤波器**：预测和校正位置，平滑跟踪结果
2. **历史记录**：保存最近几帧的偏移量
3. **中值滤波**：使用中值而非平均值减少异常值影响
4. **目标丢失处理**：当目标短暂消失时，保持发送最后一次有效的偏移值，避免发送0值

### 屏幕录制

系统包含自动屏幕录制功能：

- 程序启动时自动开始录制
- 程序关闭时自动停止录制并保存视频
- 视频保存在项目的video文件夹中，使用时间戳命名

## 使用方法

### 基本使用

1. 安装Python和必要的依赖库
2. 运行`main.py`启动系统
3. 系统将自动开始捕获摄像头图像、检测绿色光点并录制屏幕
4. 按'q'键退出程序，录制将自动停止并保存

### 权重矩阵可视化

运行`weight_matrix_visualizer.py`脚本，系统将自动生成权重矩阵的热力图可视化，并在默认浏览器中打开HTML页面展示。

## 参数调整

### HSV阈值

可以通过调整HSV阈值来适应不同的光源和环境条件：

- H（色调）：控制颜色范围（绿色通常在35-85之间）
- S（饱和度）：控制颜色的纯度（较高的值可以排除白色和灰色）
- V（亮度）：控制亮度范围（较高的值可以排除暗区域）

### 权重矩阵

可以修改`create_weight_matrix`函数中的参数来调整权重分布：

- `middle_start_y`和`middle_end_y`：控制中间区域的范围
- 权重值：调整顶部和底部区域的权重（当前为0.2）

### 录制设置

可以在`screen_recorder.py`中调整以下参数：

- `fps`：录制的帧率
- `output_folder`：视频保存路径

## 依赖库

- OpenCV (cv2)
- NumPy
- pyautogui (用于屏幕捕获)
- threading (用于异步录制)
- webbrowser (用于可视化工具)

## 注意事项

1. 系统性能受光照条件影响较大
2. 在复杂背景下可能需要调整HSV阈值
3. 权重矩阵应根据实际应用场景调整
4. 屏幕录制可能占用较多系统资源，如有性能问题可适当降低帧率
5. 连续运行可能生成大量视频文件，注意定期清理存储空间
