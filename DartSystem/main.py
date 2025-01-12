import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from ctypes import *
import time

from opencv_green_detection import create_trackbars, get_trackbar_values, detect_green_light_and_offset
from serial_communication import SerialCommunication

# 根据系统选择正确的库路径
if sys.platform.startswith("win"):
    sys.path.append("./MvImport")
    from MvImport.MvCameraControl_class import *
else:
    sys.path.append("./MvImport_Linux")
    from MvImport_Linux.MvCameraControl_class import *

def get_Value(cam, param_type="float_value", node_name=""):
    """获取相机参数"""
    if param_type == "float_value":
        stParam = MVCC_FLOATVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_FLOATVALUE))
        ret = cam.MV_CC_GetFloatValue(node_name, stParam)
        if ret != 0:
            print("获取参数 %s 失败! ret[0x%x]" % (node_name, ret))
            return None
        return stParam.fCurValue
    elif param_type == "enum_value":
        stParam = MVCC_ENUMVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_ENUMVALUE))
        ret = cam.MV_CC_GetEnumValue(node_name, stParam)
        if ret != 0:
            print("获取参数 %s 失败! ret[0x%x]" % (node_name, ret))
            return None
        return stParam.nCurValue

def set_Value(cam, param_type="float_value", node_name="", node_value=0):
    """设置相机参数"""
    if param_type == "float_value":
        ret = cam.MV_CC_SetFloatValue(node_name, node_value)
        if ret != 0:
            print("设置参数 %s 失败! ret[0x%x]" % (node_name, ret))
    elif param_type == "enum_value":
        ret = cam.MV_CC_SetEnumValue(node_name, node_value)
        if ret != 0:
            print("设置参数 %s 失败! ret[0x%x]" % (node_name, ret))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 添加帧率计算相关变量（移到最前面）
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        
        self.initUI()
        self.setup_camera()
        self.setup_serial()
        
        # 创建定时器更新图像
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 刷新一次
        
        # 创建 HSV 阈值调节滑动条
        create_trackbars()

    def setup_camera(self):
        """初始化相机"""
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("枚举设备失败! ret[0x%x]" % ret)
            return

        if deviceList.nDeviceNum == 0:
            print("没有找到设备!")
            return

        print("找到 %d 个设备!" % deviceList.nDeviceNum)

        # 创建相机实例
        self.camera = MvCamera()
        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.camera.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("创建句柄失败! ret[0x%x]" % ret)
            return

        # 打开设备
        ret = self.camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("打开设备失败! ret[0x%x]" % ret)
            return

        # 设置相机参数
        set_Value(self.camera, "float_value", "ExposureTime", 16000)  # 曝光时间
        set_Value(self.camera, "float_value", "Gain", 15.9)  # 增益值

        # 开始取流
        ret = self.camera.MV_CC_StartGrabbing()
        if ret != 0:
            print("开始取流失败! ret[0x%x]" % ret)
            return

        # 获取数据包大小
        stParam = MVCC_INTVALUE_EX()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
        ret = self.camera.MV_CC_GetIntValueEx("PayloadSize", stParam)
        if ret != 0:
            print("获取数据包大小失败! ret[0x%x]" % ret)
            return





        # 获取数据包大小
        stParam = MVCC_INTVALUE_EX()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
        ret = self.camera.MV_CC_GetIntValueEx("PayloadSize", stParam)
        if ret != 0:
            print("获取数据包大小失败! ret[0x%x]" % ret)
            return

        self.data_size = stParam.nCurValue
        self.pData = (c_ubyte * self.data_size)()
        self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(self.stFrameInfo), 0, sizeof(self.stFrameInfo))

    def read_frame(self):
        """读取一帧图像"""
        ret = self.camera.MV_CC_GetOneFrameTimeout(self.pData, self.data_size, self.stFrameInfo, 1000)
        if ret == 0:
            # 转换图像格式
            data = np.frombuffer(self.pData, dtype=np.uint8)
            
            # 根据像素格式处理图像
            frame = data.reshape((self.stFrameInfo.nHeight, self.stFrameInfo.nWidth))
            print(f"当前像素格式: {self.stFrameInfo.enPixelType}")
            
            if self.stFrameInfo.enPixelType == 17301505:  # Mono8
                img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif self.stFrameInfo.enPixelType == 17301513:  # BayerGB8
                img = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)  # 使用BG2BGR而不是GB2BGR
            elif self.stFrameInfo.enPixelType == 17301514:  # BayerRG8
                img = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            elif self.stFrameInfo.enPixelType == 17301515:  # BayerGR8
                img = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            elif self.stFrameInfo.enPixelType == 17301516:  # BayerBG8
                img = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            else:
                print(f"未知的像素格式: {self.stFrameInfo.enPixelType}，尝试其他转换方式")
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                except:
                    try:
                        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    except:
                        print("无法转换图像格式")
                        return False, None
            
            # 将BGR转换为RGB用于显示
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return True, img
        else:
            print("获取一帧图像失败: ret[0x%x]" % ret)
            return False, None

    def setup_serial(self):
        # 初始化串口通信
        try:
            self.serial_comm = SerialCommunication(port='/dev/ttyACM0', baudrate=115200)
            print("串口初始化成功")
            # 添加状态帧计数器
            self.status_frame_counter = 0
        except Exception as e:
            print(f"串口初始化失败: {str(e)}")
            self.serial_comm = None

    def update_frame(self):
        # 计算帧率
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time != 0 else 0
        self.prev_frame_time = self.curr_frame_time
        
        ret, frame = self.read_frame()
        if not ret:
            return

        # 获取滑动条的 HSV 阈值
        lower_hsv, upper_hsv = get_trackbar_values()

        # 检测是否存在绿光，并获取偏差值和轮廓信息
        green_detected, horizontal_offset, contour_info = detect_green_light_and_offset(frame, lower_hsv, upper_hsv)

        # 显示图像和掩膜
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 在原始图像上绘制检测到的区域
        result = frame.copy()
        
        # 始终绘制相机中心点和十字线
        h, w = result.shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        # 显示帧率
        cv2.putText(result, f"FPS: {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 根据检测结果发送数据
        if self.serial_comm is not None:
            try:
                # 每隔10帧发送一次状态帧
                if self.status_frame_counter % 10 == 0:
                    # 发送状态帧，参数为是否检测到目标
                    self.serial_comm.send_status_frame(green_detected)
                    status_text = "检测到目标" if green_detected else "未检测到目标"
                    print(f"发送状态帧：A5 01 {0x01 if green_detected else 0x00:02X} ({status_text})")
                    # 在图像上显示状态帧信息
                    cv2.putText(result, f"Status Frame: {status_text}", (10, 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 每帧都发送偏差值帧
                if green_detected and contour_info:
                    print(f"检测到绿光！偏差值: {horizontal_offset:d}")
                    # 发送偏差值帧
                    self.serial_comm.send_offset_frame(horizontal_offset)
                    print(f"发送偏差值帧：A5 02 {(horizontal_offset >> 8) & 0xFF:02X} {horizontal_offset & 0xFF:02X}")
                else:
                    print("未检测到绿光")
                    # 未检测到时发送0
                    self.serial_comm.send_offset_frame(0)
                    print("发送偏差值帧：A5 02 00 00")
                
                self.status_frame_counter += 1
                
            except Exception as e:
                print(f"发送数据时出错: {str(e)}")

        # 绘制中心十字线
        cv2.line(result, (center_x, 0), (center_x, h), (255, 0, 0), 1)  # 垂直线
        cv2.line(result, (0, center_y), (w, center_y), (255, 0, 0), 1)  # 水平线
        cv2.circle(result, (center_x, center_y), 5, (255, 0, 0), -1)  # 中心点
        
        if green_detected and contour_info:
            # 绘制目标边界框
            x, y, w, h = contour_info['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制轮廓
            cv2.drawContours(result, [contour_info['contour']], -1, (0, 255, 0), 2)
            
            # 绘制目标中心点
            target_center = contour_info['center']
            cv2.circle(result, target_center, 3, (0, 0, 255), -1)
            
            # 显示检测框的数据
            info_y = 110  # 起始y坐标
            cv2.putText(result, f"Area: {contour_info['area']:.0f}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, f"Size: {w}x{h}", (10, info_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, f"Center: ({target_center[0]}, {target_center[1]})", (10, info_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制从中心到目标的连线
            center = (center_x, center_y)
            target_center = contour_info['center']
            cv2.line(result, center, target_center, (0, 255, 255), 2)
            
            # 显示偏差值，添加正负号
            offset_text = f"Offset: {'+' if horizontal_offset > 0 else ''}{horizontal_offset:d}"
            cv2.putText(result, offset_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示图像
        h, w, ch = result.shape
        bytes_per_line = ch * w
        qt_image = QImage(result.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_label.setPixmap(scaled_pixmap)

        # 显示掩膜图像
        h, w, ch = mask_colored.shape
        bytes_per_line = ch * w
        qt_mask = QImage(mask_colored.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        mask_pixmap = QPixmap.fromImage(qt_mask)
        scaled_mask = mask_pixmap.scaled(self.mask_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.mask_label.setPixmap(scaled_mask)

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.camera is not None:
            self.camera.MV_CC_StopGrabbing()
            self.camera.MV_CC_CloseDevice()
            self.camera.MV_CC_DestroyHandle()
        if self.serial_comm is not None:
            self.serial_comm.close()
        event.accept()

    def initUI(self):
        """初始化UI"""
        self.setWindowTitle('相机检测')
        self.setGeometry(100, 100, 1200, 600)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()

        # 创建左侧和右侧布局
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # 创建标签显示相机图像
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.camera_label)

        # 创建标签显示掩膜图像
        self.mask_label = QLabel()
        self.mask_label.setMinimumSize(640, 480)
        right_layout.addWidget(self.mask_label)

        # 将左右布局添加到主布局
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        # 设置主布局
        main_widget.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())