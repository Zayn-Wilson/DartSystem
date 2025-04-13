import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from ctypes import *
import time
import atexit

from opencv_green_detection import create_trackbars, get_trackbar_values, detect_green_light_and_offset, detector
from serial_communication import SerialCommunication
# from screen_recorder import start_screen_recording, stop_screen_recording
from system_recorder import start_system_recording as start_screen_recording
from system_recorder import stop_system_recording as stop_screen_recording
from serial_receiver import SerialReceiver  # 导入SerialReceiver类
#改动
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
        
        # 初始化串口数据相关变量
        self.received_yaw = 0
        self.received_preload = 0
        self.last_serial_update_time = 0  # 确保变量被正确初始化
        self.serial_debug_counter = 0  # 用于追踪串口读取次数
        
        self.initUI()
        self.setup_camera()
        self.setup_serial()
        self.setup_serial_receiver()  # 设置串口接收
        
        # 创建定时器更新图像
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 刷新一次
        
        # 创建定时器读取串口数据 - 使用更快的更新频率
        self.serial_timer = QTimer()
        self.serial_timer.timeout.connect(self.read_serial_data)
        self.serial_timer.start(10)  # 10ms 读取一次串口数据
        
        print("定时器已启动: 图像刷新-30ms, 串口读取-10ms")
        
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
            # 尝试连接电控的串口
            self.serial_comm = SerialCommunication(port='/dev/my_stm32', baudrate=115200)
            # 如果连接失败，可以尝试其他常用端口
            print("串口初始化成功，使用端口：/dev/my_stm32")
            
            # 添加状态帧计数器
            self.status_frame_counter = 0
            
            # 检查串口是否已打开
            if not self.serial_comm.serial.is_open:
                self.serial_comm.serial.open()
                print("重新打开串口")
            
            # 清空缓冲区，确保获取最新数据
            self.serial_comm.serial.reset_input_buffer()
            self.serial_comm.serial.reset_output_buffer()
            
            # 注释掉接收数据的尝试
            # self.serial_comm.check_for_data()
            
        except Exception as e:
            print(f"串口初始化失败: {str(e)}")
            
            # 尝试备用端口
            try:
                self.serial_comm = SerialCommunication(port='/dev/ttyACM0', baudrate=115200)
                print("使用备用串口初始化成功，使用端口：/dev/ttyACM0")
            except Exception as e2:
                print(f"备用串口初始化也失败: {str(e2)}")
                self.serial_comm = None

    def setup_serial_receiver(self):
        """设置串口接收器"""
        try:
            # 使用ttyACM1端口，与serial_receiver.py默认值一致
            self.serial_receiver = SerialReceiver(port='/dev/ttyACM1', baudrate=115200)
            print("串口接收器初始化成功，使用端口：/dev/ttyACM1")
            
            # 简单测试是否能正常收到数据
            for _ in range(3):
                test_data = self.serial_receiver.read_frame()
                if test_data is not None:
                    print(f"测试数据读取成功: yaw={test_data['yaw']}, preload={test_data['preload']}")
                    # 立即更新显示数据
                    self.received_yaw = test_data['yaw']
                    self.received_preload = test_data['preload']
                    self.last_serial_update_time = time.time()
                    break
                time.sleep(0.1)
            
        except Exception as e:
            print(f"串口接收器初始化失败: {str(e)}")
            # 尝试备用端口 - 符号链接
            try:
                self.serial_receiver = SerialReceiver(port='/dev/my_stm32', baudrate=115200)
                print("串口接收器初始化成功，使用备用端口：/dev/my_stm32")
            except Exception as e2:
                print(f"备用串口接收器初始化也失败: {str(e2)}")
                self.serial_receiver = None

    def read_serial_data(self):
        """读取串口数据"""
        self.serial_debug_counter += 1
        # 大幅降低定时器触发次数的打印频率
        # if self.serial_debug_counter % 500 == 0: 
        #     print(f"串口读取定时器触发次数: {self.serial_debug_counter}")

        if self.serial_receiver is not None:
            try:
                if not self.serial_receiver.serial.is_open:
                    # 保留串口重连逻辑和打印
                    print("接收串口已关闭，尝试重新打开...")
                    try:
                        self.serial_receiver.serial.open()
                        print("接收串口重新打开成功。")
                    except Exception as open_err:
                        print(f"重新打开接收串口失败: {open_err}")
                        return 

                # # 暂时移除缓冲区检查和重置的打印
                # if self.serial_debug_counter % 50 == 1: 
                #      try:
                #          bytes_in_waiting = self.serial_receiver.serial.in_waiting
                #          print(f"[调试] 输入缓冲区字节数: {bytes_in_waiting}")
                #      except Exception as e:
                #          print(f"[调试] 无法检查缓冲区: {e}")
                # if self.serial_debug_counter % 50 == 0:
                #     # print("[调试] 尝试重置输入缓冲区...")
                #     self.serial_receiver.serial.reset_input_buffer()
                #     # print("[调试] 输入缓冲区已重置。")

                data = self.serial_receiver.read_frame()

                # 降低 read_frame 返回值的打印频率
                if self.serial_debug_counter % 50 == 0: # 每 50 次打印一次 (约 0.5s)
                    print(f"[调试] read_frame 返回: {data}")

                if data is not None:
                    self.last_serial_update_time = time.time()

                    if self.received_yaw != data['yaw'] or self.received_preload != data['preload']:
                        # 保留数据更新的打印，因为这比较关键
                        print(f"数据更新: yaw: {self.received_yaw} -> {data['yaw']}, preload: {self.received_preload} -> {data['preload']}")
                        self.received_yaw = data['yaw']
                        self.received_preload = data['preload']
                    
                    # 降低当前值的打印频率
                    # if self.serial_debug_counter % 100 == 0: # 每 100 次打印一次 (约 1s)
                    #     print(f"当前串口数据: yaw={self.received_yaw}, preload={self.received_preload}")

            except Exception as e:
                # 保留错误处理和重连逻辑及打印
                print(f"读取串口数据错误: {str(e)}")
                print("尝试重新初始化串口接收器...")
                try:
                    # ... (重连逻辑保持不变) ...
                    if self.serial_receiver:
                        self.serial_receiver.close()
                    time.sleep(0.5)
                    primary_port = '/dev/ttyACM1'
                    backup_port = '/dev/my_stm32'
                    try:
                        self.serial_receiver = SerialReceiver(port=primary_port, baudrate=115200)
                        print(f"串口接收器在 {primary_port} 重新初始化成功")
                    except Exception:
                         print(f"主端口 {primary_port} 失败, 尝试备用端口 {backup_port}...")
                         try:
                             self.serial_receiver = SerialReceiver(port=backup_port, baudrate=115200)
                             print(f"串口接收器在 {backup_port} 重新初始化成功")
                         except Exception as reinit_err_backup:
                              print(f"备用串口接收器初始化也失败: {str(reinit_err_backup)}")
                              self.serial_receiver = None 

                except Exception as reinit_err:
                    print(f"串口接收器重新初始化过程中出错: {str(reinit_err)}")
                    self.serial_receiver = None 

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
        
        # 设置文字显示的基础位置和间距
        text_start_x = 10
        text_start_y = 30  # 起始位置调高
        line_spacing = 40  # 增加行间距
        
        # 显示帧率
        fps_y = text_start_y
        cv2.putText(result, f"FPS: {fps:.1f}", 
                    (text_start_x, fps_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 计算偏移值（整数）
        offset_int = int(horizontal_offset) if green_detected and contour_info else int(detector.last_valid_offset)
        
        # 显示偏移值
        offset_y = fps_y + line_spacing
        cv2.putText(result, f"Offset: {'+' if offset_int > 0 else ''}{offset_int:d}", 
                    (text_start_x, offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 如果检测到目标，显示面积
        if green_detected and contour_info:
            area = contour_info['area']
            area_y = offset_y + line_spacing
            cv2.putText(result, f"Area: {area:.0f}", 
                        (text_start_x, area_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示接收到的串口数据
        serial_data_y = (area_y if green_detected and contour_info else offset_y) + line_spacing
        
        # 计算自上次接收数据以来的时间
        time_since_update = time.time() - self.last_serial_update_time
        
        # 如果在最近2秒内有更新，则用绿色显示；否则用红色显示
        serial_color = (0, 255, 0) if time_since_update < 2.0 else (0, 0, 255)
        
        cv2.putText(result, f"Received Yaw: {int(self.received_yaw)}", 
                    (text_start_x, serial_data_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, serial_color, 2)
                    
        cv2.putText(result, f"Received Preload: {int(self.received_preload)}", 
                    (text_start_x, serial_data_y + line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, serial_color, 2)
            
        # 恢复串口通信功能
        if self.serial_comm is not None:
            try:
                # 将布尔值明确转换为整数
                # 只有当检测到绿灯时，status_int和green_status_int才为1
                status_int = 1 if green_detected and contour_info else 0
                green_status_int = 1 if green_detected and contour_info else 0
                
                # 发送合并帧
                self.serial_comm.send_combined_frame(status_int, green_status_int, offset_int)
                
                # 显示发送信息（仅在控制台）
                status_text = "检测到目标" if status_int == 1 else "未检测到目标"
                green_status_text = "检测到绿灯" if green_status_int == 1 else "未检测到绿灯"
                
                print(f"发送合并帧：A5 {status_int:02X} {green_status_int:02X} {(offset_int >> 8) & 0xFF:02X} {offset_int & 0xFF:02X}")
                print(f"  - 状态: {status_text}")
                print(f"  - 绿灯: {green_status_text}")
                print(f"  - 偏移值: {offset_int:d}")
                
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
            
            # 绘制从中心到目标的连线
            center = (center_x, center_y)
            cv2.line(result, center, target_center, (0, 255, 255), 2)

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
        print("正在关闭应用...")
        
        # 停止定时器
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'serial_timer'):
            self.serial_timer.stop()
        
        # 关闭相机
        if self.camera is not None:
            print("关闭相机...")
            self.camera.MV_CC_StopGrabbing()
            self.camera.MV_CC_CloseDevice()
            self.camera.MV_CC_DestroyHandle()
        
        # 停止屏幕录制
        print("停止屏幕录制...")
        stop_screen_recording()
        
        # 关闭串口
        if self.serial_comm is not None:
            print("关闭发送串口...")
            self.serial_comm.close()
        
        if hasattr(self, 'serial_receiver') and self.serial_receiver is not None:
            print("关闭接收串口...")
            self.serial_receiver.close()
        
        print("应用已关闭")
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

def main():
    # 启动屏幕录制
    start_screen_recording(format="webm", capture_method="xcb")  # 使用WebM格式和XCB捕获方法，确保兼容性
    
    # 注册程序退出时的回调，确保录制停止
    atexit.register(stop_screen_recording)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置HSV阈值
    lower_hsv = np.array([35, 50, 50])
    upper_hsv = np.array([85, 255, 255])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测绿光并获取偏移值
        detected, offset, contour_info = detector.detect_green_light_and_offset(frame, lower_hsv, upper_hsv)
        
        # 在画面上显示偏移值
        cv2.putText(frame, f"Offset: {offset}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 如果检测到目标，绘制边界框和中心点
        if detected and contour_info:
            x, y, w, h = contour_info['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, contour_info['center'], 5, (0, 0, 255), -1)
        
        # 显示处理后的画面
        cv2.imshow('Green Light Detection', frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 启动屏幕录制
    # 使用系统录制功能，不会导致屏幕闪烁
    start_screen_recording(format="webm", capture_method="xcb")  # 使用WebM格式和XCB捕获方法，确保兼容性
    
    # 注册程序退出时的回调，确保录制停止
    atexit.register(stop_screen_recording)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())