import cv2
import numpy as np
from collections import deque

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

    def create_weight_matrix(self, frame):
        height, width = frame.shape[:2]
        weight_matrix = np.ones((height, width))
        
        # 定义中间区域的范围
        middle_start_y = int(height * 0.35)  # 从35%开始
        middle_end_y = int(height * 0.65)    # 到65%结束
        
        # 削弱顶部和底部区域
        weight_matrix[:middle_start_y, :] *= 0.2  # 顶部区域权重降低到0.2
        weight_matrix[middle_end_y:, :] *= 0.2    # 底部区域权重降低到0.2
        
        # 中间区域保持完整权重
        weight_matrix[middle_start_y:middle_end_y, :] = 1.0
        
        return weight_matrix

    def detect_green_light_and_offset(self, frame, lower_hsv, upper_hsv):
        """
        检测图像中的绿光并计算偏差值，使用多种滤波方法提高稳定性
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # 使用更小的核进行高斯模糊
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        
        # 创建掩膜
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # 调整形态学操作的核大小
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # 形态学操作
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 降低面积阈值
        MIN_AREA = 50  # 降低最小面积阈值
        MAX_AREA = 15000  # 增大最大面积阈值
        
        valid_contours = []
        
        # 收集所有有效的轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_AREA <= area <= MAX_AREA:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # 简化权重计算，只考虑面积
                    valid_contours.append({
                        'contour': contour,
                        'area': area,
                        'center': (cx, cy),
                        'weight': area  # 使用面积作为权重
                    })
        
        if valid_contours:
            # 选择面积最大的轮廓
            best_contour = max(valid_contours, key=lambda x: x['area'])
            
            max_contour = best_contour['contour']
            area = best_contour['area']
            cx, cy = best_contour['center']
            
            x, y, w, h = cv2.boundingRect(max_contour)
            
            frame_center_x = frame.shape[1] // 2
            horizontal_offset = cx - frame_center_x
            
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
                'smoothed_offset': smoothed_offset
            }
            
            return True, smoothed_offset, contour_info

        # 目标丢失处理
        self.lost_count += 1
        if self.lost_count < self.MAX_LOST_FRAMES:
            predicted_offset = self.kalman_filter.update(None)
            return True, int(predicted_offset), None
        
        self.offset_history.clear()
        return False, 0, None

# 创建检测器实例
detector = GreenLightDetector()

# 修改原来的函数为包装函数
def detect_green_light_and_offset(frame, lower_hsv, upper_hsv):
    return detector.detect_green_light_and_offset(frame, lower_hsv, upper_hsv)

def create_trackbars():
    """
    创建滑动条，用于调节 HSV 阈值
    """
    cv2.namedWindow('HSV Trackbars')
    # 调整默认值以更好地检测绿光
    cv2.createTrackbar('H min', 'HSV Trackbars', 35, 179, lambda x: None)
    cv2.createTrackbar('H max', 'HSV Trackbars', 85, 179, lambda x: None)
    cv2.createTrackbar('S min', 'HSV Trackbars', 50, 255, lambda x: None)
    cv2.createTrackbar('S max', 'HSV Trackbars', 255, 255, lambda x: None)
    cv2.createTrackbar('V min', 'HSV Trackbars', 50, 255, lambda x: None)
    cv2.createTrackbar('V max', 'HSV Trackbars', 255, 255, lambda x: None)

def get_trackbar_values():
    """
    获取滑动条的 HSV 阈值
    :return: lower_hsv, upper_hsv
    """
    h_min = cv2.getTrackbarPos('H min', 'HSV Trackbars')
    h_max = cv2.getTrackbarPos('H max', 'HSV Trackbars')
    s_min = cv2.getTrackbarPos('S min', 'HSV Trackbars')
    s_max = cv2.getTrackbarPos('S max', 'HSV Trackbars')
    v_min = cv2.getTrackbarPos('V min', 'HSV Trackbars')
    v_max = cv2.getTrackbarPos('V max', 'HSV Trackbars')
    
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])
    
    return lower_hsv, upper_hsv