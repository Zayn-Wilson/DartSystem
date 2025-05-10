import cv2
import numpy as np
from collections import deque
#识别
# 全局变量声明
GUI_AVAILABLE = False  # 默认为False，将在create_trackbars中尝试创建窗口时更新
DEFAULT_HSV_VALUES = {
    'H min': 30, 'H max': 90, 
    'S min': 30, 'S max': 255, 
    'V min': 120, 'V max': 255
}

class KalmanFilter1D:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(2, 1)
        self.kalman.measurementMatrix = np.array([[1., 0.]], np.float32)
        self.kalman.transitionMatrix = np.array([[1., 1.], [0., 1.]], np.float32)
        self.kalman.processNoiseCov = np.array([[1e-3, 0.], [0., 1e-3]], np.float32)
        self.kalman.measurementNoiseCov = np.array([[1e-2]], np.float32)
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            if measurement is None:
                return 0  # 如果未初始化且没有测量值，返回0
            self.kalman.statePre = np.array([[measurement], [0.]], np.float32)
            self.initialized = True
            return measurement

        prediction = self.kalman.predict()
        if measurement is not None:
            correction = self.kalman.correct(np.array([[measurement]], np.float32))
            return correction[0, 0]
        return prediction[0, 0]

class GreenLightDetector:
    def __init__(self, history_size=3):
        self.offset_history = deque(maxlen=history_size)
        self.kalman_filter = KalmanFilter1D()
        self.last_valid_offset = 0
        self.lost_count = 0
        self.MAX_LOST_FRAMES = 5
        self.weight_matrix = None
        # 增加最近有效检测的历史记录
        self.recent_detections = deque(maxlen=10)
        # 添加亮度阈值
        self.brightness_threshold = 200
        # 保存上一帧检测到的位置信息
        self.last_valid_position = None
        # 初始化有效HSV计数器
        self.hsv_range_samples = []
        self.max_samples = 100
        # 预览掩码和结果
        self.debug_mask = None
        self.debug_result = None

    def create_weight_matrix(self, frame):
        height, width = frame.shape[:2]
        weight_matrix = np.ones((height, width))
        
        # 更加强调上半部分区域
        # 将图像分成四个区域：顶部(0-25%)、上中部(25-50%)、下中部(50-75%)、底部(75-100%)
        top_end = int(height * 0.25)          # 顶部区域结束点
        upper_mid_end = int(height * 0.5)     # 上中部结束点
        lower_mid_end = int(height * 0.75)    # 下中部结束点
        
        # 大幅提高上中部权重，适度提高顶部权重，降低下部权重
        weight_matrix[:top_end, :] *= 1.5              # 顶部权重提高到1.5
        weight_matrix[top_end:upper_mid_end, :] *= 2.0 # 上中部权重提高到2.0（最高优先级）
        weight_matrix[upper_mid_end:lower_mid_end, :] *= 0.5 # 下中部降低到0.5
        weight_matrix[lower_mid_end:, :] *= 0.2        # 底部权重最低0.2
        
        return weight_matrix

    def analyze_hsv_sample(self, hsv_region):
        """分析HSV区域样本，自动优化HSV阈值"""
        if hsv_region is not None and hsv_region.size > 0:
            # 仅保留有限数量的样本，先进先出
            self.hsv_range_samples.append(hsv_region)
            if len(self.hsv_range_samples) > self.max_samples:
                self.hsv_range_samples.pop(0)
            
            # 计算所有样本的统计数据
            if len(self.hsv_range_samples) > 5:  # 至少需要5个样本才开始分析
                all_samples = np.vstack(self.hsv_range_samples)
                
                # 计算H、S、V通道的均值和标准差
                h_mean = np.mean(all_samples[:, 0])
                h_std = np.std(all_samples[:, 0])
                s_mean = np.mean(all_samples[:, 1])
                s_std = np.std(all_samples[:, 1])
                v_mean = np.mean(all_samples[:, 2])
                v_std = np.std(all_samples[:, 2])
                
                # 使用2个标准差覆盖约95%的样本
                h_min = max(0, h_mean - 2 * h_std)
                h_max = min(179, h_mean + 2 * h_std)
                s_min = max(0, s_mean - 2 * s_std)
                s_max = min(255, s_mean + 2 * s_std)
                v_min = max(0, v_mean - 2 * v_std)
                v_max = min(255, v_mean + 2 * v_std)
                
                # 返回优化的HSV范围
                return (
                    np.array([int(h_min), int(s_min), int(v_min)]),
                    np.array([int(h_max), int(s_max), int(v_max)])
                )
        return None, None

    def is_valid_green_target(self, contour, hsv, original_frame):
        """增强的目标有效性验证"""
        # 基础几何特征检查
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 1. 计算圆形度 (4π*面积/周长²)，完美圆形为1
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 2. 拟合椭圆并检查长宽比
        if len(contour) >= 5:  # 至少需要5个点才能拟合椭圆
            ellipse = cv2.fitEllipse(contour)
            (_, _), (width, height), _ = ellipse
            aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
        else:
            aspect_ratio = 1.0  # 点太少，假设是圆形
        
        # 3. 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 检查位置 - 更倾向于上半部分的目标
        img_height = original_frame.shape[0]
        position_score = 1.0
        if y < img_height * 0.5:  # 在上半部分
            position_score = 2.0   # 位置评分加倍
        
        # 4. 在原始帧中提取ROI
        roi = original_frame[y:y+h, x:x+w]
        
        # 5. 分析ROI的亮度特征
        if roi.size > 0:  # 确保ROI不为空
            # 转为灰度
            if len(roi.shape) == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            else:
                roi_gray = roi
                
            # 计算亮度统计
            mean_brightness = np.mean(roi_gray)
            max_brightness = np.max(roi_gray)
            brightness_std = np.std(roi_gray)
            
            # 6. 创建目标区域的HSV掩码
            mask = np.zeros_like(original_frame[:,:,0])
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # 7. 获取掩码内的HSV值
            hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask)
            non_zero = hsv_masked[mask > 0]
            
            if non_zero.size > 0:
                # 保存有效的HSV样本用于自适应
                self.analyze_hsv_sample(non_zero)
                
                # 计算色相的集中度
                h_values = non_zero[:, 0]
                h_std = np.std(h_values)
                
                # 亮度与饱和度比例
                v_values = non_zero[:, 2]
                s_values = non_zero[:, 1]
                sv_ratio = np.mean(s_values) / np.mean(v_values) if np.mean(v_values) > 0 else 0
                
                # 亮度稳定性，真实LED灯更均匀
                brightness_uniformity = 1 - (brightness_std / (mean_brightness + 1e-5))
                
                # 计算绿色指标 - 真实绿灯在HSV中H值应该在绿色范围(约45-75)
                h_mean = np.mean(h_values)
                green_score = 0
                if 30 <= h_mean <= 90:  # 扩大绿色范围
                    # 45-75是最佳绿色范围
                    if 40 <= h_mean <= 80:
                        green_score = 1.0
                    else:
                        green_score = 0.7  # 30-40或80-90是次优绿色
                
                # 放宽判断标准
                is_valid = (
                    circularity > 0.4 and              # 进一步降低圆形度要求
                    aspect_ratio > 0.4 and             # 进一步降低长宽比要求
                    mean_brightness > 70 and           # 进一步降低亮度要求
                    brightness_std < 80 and            # 进一步放宽亮度均匀性要求
                    h_std < 40 and                     # 进一步放宽色相一致性要求
                    sv_ratio > 0.2 and sv_ratio < 4.0 and  # 进一步放宽饱和度与亮度比例要求
                    green_score > 0.3                  # 进一步降低绿色评分要求
                )
                
                # 计算综合得分 - 用于后续权重计算
                score = (
                    circularity * 1.5 +                    # 降低圆形度权重
                    aspect_ratio * 0.8 +                   # 降低长宽比权重
                    brightness_uniformity * 1.2 +          # 降低亮度均匀性权重
                    (1.0 - min(h_std / 40.0, 1.0)) * 0.8 + # 放宽色相一致性要求
                    green_score * 2.0 +                    # 保持绿色评分权重
                    position_score * 2.5                   # 增加位置评分权重
                )
                
                return is_valid, {
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'mean_brightness': mean_brightness,
                    'brightness_std': brightness_std,
                    'h_std': h_std,
                    'h_mean': h_mean,
                    'sv_ratio': sv_ratio,
                    'position_score': position_score,
                    'green_score': green_score,
                    'combined_score': score
                }
        
        return False, {}

    def detect_circles(self, mask, min_radius=5, max_radius=150):
        """
        使用霍夫圆变换检测圆形，调整参数以适应更大的圆
        """
        # 使用霍夫圆变换，降低参数阈值以检测更多圆
        circles = cv2.HoughCircles(
            mask, 
            cv2.HOUGH_GRADIENT, 
            dp=1,               # 累加器分辨率与图像分辨率的比率
            minDist=30,         # 减小检测到的圆的最小距离
            param1=40,          # 降低Canny边缘检测的高阈值
            param2=20,          # 降低累加器阈值，更容易检测到不完美的圆
            minRadius=min_radius,
            maxRadius=max_radius # 增大最大半径
        )
        
        return circles

    def detect_green_light_and_offset(self, frame, lower_hsv, upper_hsv):
        """
        检测图像中的绿光并计算偏差值，使用多种滤波方法提高稳定性
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # 高斯模糊预处理，减少噪声 - 使用更大的核来处理模糊的光源
        hsv = cv2.GaussianBlur(hsv, (7, 7), 0)
        
        # 创建掩膜
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # 保存原始掩码用于调试
        self.debug_mask = mask.copy()
        
        # 调整形态学操作的核大小 - 使用更大的核来处理空心圆
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 增大闭运算核
        
        # 形态学操作 - 先闭运算填充空心，再开运算去除噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # 添加额外的闭运算确保填充空心
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 调整面积阈值 - 进一步放宽
        MIN_AREA = 30    # 进一步降低最小面积阈值
        MAX_AREA = 20000 # 进一步增大最大面积阈值
        
        valid_contours = []
        
        # 创建或更新权重矩阵
        if self.weight_matrix is None or self.weight_matrix.shape[:2] != frame.shape[:2]:
            self.weight_matrix = self.create_weight_matrix(frame)
        
        # 对于调试，创建结果图像
        debug_img = frame.copy()
        
        # 收集所有有效的轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_AREA <= area <= MAX_AREA:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 应用增强的目标验证
                    is_valid, metrics = self.is_valid_green_target(contour, hsv, frame)
                    
                    # 在调试图像上绘制所有检测到的轮廓
                    cv2.drawContours(debug_img, [contour], -1, (0, 0, 255), 1)
                    
                    if is_valid:
                        # 应用权重矩阵 - 基于位置
                        position_weight = self.weight_matrix[cy, cx]
                        
                        # 计算位置连续性分数 - 如果与上一个有效位置接近，得分更高
                        continuity_score = 1.0
                        if self.last_valid_position is not None:
                            last_x, last_y = self.last_valid_position
                            dist = np.sqrt((cx - last_x) ** 2 + (cy - last_y) ** 2)
                            max_dist = frame.shape[0] * 0.3  # 最大距离设为图像高度的30%
                            if dist < max_dist:
                                continuity_score = 1.0 + (1.0 - dist / max_dist)  # 1.0~2.0之间
                        
                        # 计算最终权重 - 综合考虑面积、位置权重、目标评分和位置连续性
                        combined_score = metrics.get('combined_score', 1.0)
                        final_weight = area * position_weight * combined_score * continuity_score
                        
                        # 在调试图像上标记有效目标
                        cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
                        cv2.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)
                        
                        valid_contours.append({
                            'contour': contour,
                            'area': area,
                            'center': (cx, cy),
                            'weight': final_weight,
                            'metrics': metrics
                        })
        
        # 如果常规轮廓检测失败，尝试霍夫圆检测
        if not valid_contours:
            # 先尝试检测小圆（优先级更高）
            small_circles = self.detect_circles(mask, min_radius=5, max_radius=50)
            
            if small_circles is not None:
                circles = np.uint16(np.around(small_circles))
                for i in circles[0, :]:
                    # 圆心坐标和半径
                    cx, cy, radius = i[0], i[1], i[2]
                    
                    # 创建圆形轮廓
                    circle_contour = []
                    for angle in range(0, 360, 5):  # 每5度一个点
                        x = cx + int(radius * np.cos(angle * np.pi / 180))
                        y = cy + int(radius * np.sin(angle * np.pi / 180))
                        circle_contour.append([[x, y]])
                    
                    circle_contour = np.array(circle_contour, dtype=np.int32)
                    
                    # 计算面积
                    area = np.pi * radius * radius
                    
                    if MIN_AREA <= area <= MAX_AREA:
                        # 在调试图像上绘制检测到的圆
                        cv2.circle(debug_img, (cx, cy), radius, (0, 255, 0), 2)
                        cv2.circle(debug_img, (cx, cy), 2, (0, 0, 255), 3)
                        
                        # 添加到有效轮廓列表
                        valid_contours.append({
                            'contour': circle_contour,
                            'area': area,
                            'center': (cx, cy),
                            'weight': area * 2.5,  # 给小圆更高的权重
                            'metrics': {'circularity': 1.0, 'green_score': 0.9, 'combined_score': 5.5}  # 提高小圆的评分
                        })
            
            # 如果没有检测到小圆，再尝试检测大圆
            if not valid_contours:
                large_circles = self.detect_circles(mask, min_radius=30, max_radius=150)
                
                if large_circles is not None:
                    circles = np.uint16(np.around(large_circles))
                    for i in circles[0, :]:
                        # 圆心坐标和半径
                        cx, cy, radius = i[0], i[1], i[2]
                        
                        # 创建圆形轮廓
                        circle_contour = []
                        for angle in range(0, 360, 5):  # 每5度一个点
                            x = cx + int(radius * np.cos(angle * np.pi / 180))
                            y = cy + int(radius * np.sin(angle * np.pi / 180))
                            circle_contour.append([[x, y]])
                        
                        circle_contour = np.array(circle_contour, dtype=np.int32)
                        
                        # 计算面积
                        area = np.pi * radius * radius
                        
                        # 不检查最小面积，只检查最大面积
                        if area <= MAX_AREA * 1.5:  # 为大圆提供更宽松的最大面积限制
                            # 在调试图像上绘制检测到的圆
                            cv2.circle(debug_img, (cx, cy), radius, (0, 255, 0), 2)
                            cv2.circle(debug_img, (cx, cy), 2, (0, 0, 255), 3)
                            
                            # 添加到有效轮廓列表，给大圆更高的权重
                            valid_contours.append({
                                'contour': circle_contour,
                                'area': area,
                                'center': (cx, cy),
                                'weight': area * 3,  # 给大圆更高的权重
                                'metrics': {'circularity': 1.0, 'green_score': 0.9, 'combined_score': 6.0}  # 完美圆形
                            })
        
        self.debug_result = debug_img
        
        if valid_contours:
            # 给小圆额外的权重提升
            for i, contour_info in enumerate(valid_contours):
                area = contour_info['area']
                # 如果是小圆（面积较小），给予额外权重
                if area < 2000:  # 调整这个阈值以适应您的小圆大小
                    valid_contours[i]['weight'] *= 1.5  # 额外提升50%权重
            
            # 选择权重最大的轮廓
            best_contour = max(valid_contours, key=lambda x: x['weight'])
            
            max_contour = best_contour['contour']
            area = best_contour['area']
            cx, cy = best_contour['center']
            
            # 更新最后有效位置
            self.last_valid_position = (cx, cy)
            
            # 额外检查：在高亮度环境下，增加严格度
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            avg_brightness = np.mean(gray)
            
            # 如果整体亮度高，应用更严格的验证
            if avg_brightness > 180:  # 提高亮度阈值
                metrics = best_contour['metrics']
                
                # 在高亮度环境下额外检查
                if metrics.get('brightness_std', 100) > 60 or metrics.get('green_score', 0) < 0.6:
                    # 高亮度环境下不满足条件，放弃此次检测
                    self.lost_count += 1
                    if self.lost_count < self.MAX_LOST_FRAMES:
                        predicted_offset = self.kalman_filter.update(None)
                        if predicted_offset is None:
                            return True, self.last_valid_offset, None
                        return True, int(predicted_offset), None
                    return True, self.last_valid_offset, None
            
            x, y, w, h = cv2.boundingRect(max_contour)
            
            frame_center_x = frame.shape[1] // 2
            horizontal_offset = cx - frame_center_x
            
            # 添加到最近检测历史
            self.recent_detections.append((cx, cy, area))
            
            # 使用卡尔曼滤波
            filtered_offset = self.kalman_filter.update(horizontal_offset)
            
            # 添加到历史记录
            self.offset_history.append(filtered_offset)
            
            # 使用中值滤波代替平均值，减少异常值影响
            smoothed_offset = int(np.median(self.offset_history))
            
            self.last_valid_offset = smoothed_offset
            self.lost_count = 0
            
            contour_info = {
                'contour': max_contour,
                'bbox': (x, y, w, h),
                'center': (int(cx), int(cy)),
                'area': area,
                'raw_offset': horizontal_offset,
                'filtered_offset': filtered_offset,
                'smoothed_offset': smoothed_offset,
                'metrics': best_contour['metrics']
            }
            
            return True, smoothed_offset, contour_info

        # 目标丢失处理
        self.lost_count += 1
        if self.lost_count < self.MAX_LOST_FRAMES:
            # 使用卡尔曼滤波预测值
            predicted_offset = self.kalman_filter.update(None)
            # 返回检测结果和偏移
            if predicted_offset is None:
                return True, self.last_valid_offset, None  # 当predicted_offset为None时返回最后一次有效的偏差值
            return True, int(predicted_offset), None
        
        # 超过MAX_LOST_FRAMES，仍然返回最后一次有效的偏移值
        return True, self.last_valid_offset, None

# 创建检测器实例
detector = GreenLightDetector()

# 修改原来的函数为包装函数
def detect_green_light_and_offset(frame, lower_hsv, upper_hsv):
    return detector.detect_green_light_and_offset(frame, lower_hsv, upper_hsv)

def create_trackbars():
    """
    创建滑动条，用于调节 HSV 阈值
    如果无法创建窗口，将使用默认值
    """
    global GUI_AVAILABLE
    try:
        cv2.namedWindow('HSV Trackbars')
        # 根据图像中的绿光特征调整HSV范围 - 更宽松的范围
        cv2.createTrackbar('H min', 'HSV Trackbars', 30, 179, lambda x: None)  # 进一步降低H下限
        cv2.createTrackbar('H max', 'HSV Trackbars', 90, 179, lambda x: None)  # 进一步提高H上限
        cv2.createTrackbar('S min', 'HSV Trackbars', 30, 255, lambda x: None)  # 进一步降低S下限
        cv2.createTrackbar('S max', 'HSV Trackbars', 255, 255, lambda x: None)
        cv2.createTrackbar('V min', 'HSV Trackbars', 120, 255, lambda x: None)  # 调整V下限
        cv2.createTrackbar('V max', 'HSV Trackbars', 255, 255, lambda x: None)
        GUI_AVAILABLE = True
    except cv2.error:
        print("[INFO] OpenCV GUI不可用，使用默认HSV值")
        GUI_AVAILABLE = False

def get_trackbar_values():
    """
    获取滑动条的 HSV 阈值
    如果GUI不可用，则返回默认值
    :return: lower_hsv, upper_hsv
    """
    if GUI_AVAILABLE:
        try:
            h_min = cv2.getTrackbarPos('H min', 'HSV Trackbars')
            h_max = cv2.getTrackbarPos('H max', 'HSV Trackbars')
            s_min = cv2.getTrackbarPos('S min', 'HSV Trackbars')
            s_max = cv2.getTrackbarPos('S max', 'HSV Trackbars')
            v_min = cv2.getTrackbarPos('V min', 'HSV Trackbars')
            v_max = cv2.getTrackbarPos('V max', 'HSV Trackbars')
        except cv2.error:
            # 如果获取轨迹条失败，使用默认值
            h_min = DEFAULT_HSV_VALUES['H min']
            h_max = DEFAULT_HSV_VALUES['H max']
            s_min = DEFAULT_HSV_VALUES['S min']
            s_max = DEFAULT_HSV_VALUES['S max']
            v_min = DEFAULT_HSV_VALUES['V min']
            v_max = DEFAULT_HSV_VALUES['V max']
    else:
        # 使用默认值
        h_min = DEFAULT_HSV_VALUES['H min']
        h_max = DEFAULT_HSV_VALUES['H max']
        s_min = DEFAULT_HSV_VALUES['S min']
        s_max = DEFAULT_HSV_VALUES['S max']
        v_min = DEFAULT_HSV_VALUES['V min']
        v_max = DEFAULT_HSV_VALUES['V max']
    
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])
    
    return lower_hsv, upper_hsv

def get_debug_images():
    """
    获取调试图像
    :return: mask图像, 结果图像
    """
    return detector.debug_mask, detector.debug_result