import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
from ctypes import *
import time
import atexit
import serial
import struct

from opencv_green_detection import create_trackbars, get_trackbar_values, detect_green_light_and_offset, detector
# from serial_communication import SerialCommunication
# from screen_recorder import start_screen_recording, stop_screen_recording
from system_recorder import start_system_recording as start_screen_recording
from system_recorder import stop_system_recording as stop_screen_recording
from serial_receiver import SerialReceiver  # 确保导入SerialReceiver类
#改动
# 根据系统选择正确的库路径

if sys.platform.startswith("win"):
    sys.path.append("./MvImport")
    from MvImport.MvCameraControl_class import *
else:
    sys.path.append("./MvImport_Linux")
    from MvImport_Linux.MvCameraControl_class import *

# 串口通信类
class SerialCommunication(QObject):
    data_received = pyqtSignal(float, float)  # 定义信号，传递 yaw 和 preload
    connection_status = pyqtSignal(bool)  # 定义信号，传递连接状态

    def __init__(self, port='/dev/my_stm32', baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.buffer = bytearray()
        self.running = True
        self.last_yaw = 0  # 添加last_yaw变量，与SerialReceiver保持一致
        self.last_preload = 0  # 添加last_preload变量，与SerialReceiver保持一致
        self.connect_serial()

    def connect_serial(self):
        """尝试连接串口，失败时进行重试"""
        try:
            if self.serial is None or not self.serial.is_open:
                self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
                print(f"串口 {self.port} 连接成功")
                self.connection_status.emit(True)  # 连接成功
        except serial.SerialException as e:
            print(f"无法连接串口 {self.port}: {e}")
            self.connection_status.emit(False)  # 连接失败
            self.serial = None

    def read_frame(self):
        """读取并解析串口数据"""
        if self.serial and self.serial.is_open:
            try:
                while self.running:
                    if self.serial.in_waiting > 0:
                        new_data = self.serial.read(self.serial.in_waiting)
                        self.buffer.extend(new_data)
                        print(f"[接收到原始字节流] {new_data.hex()}")  # 添加方括号

                    if len(self.buffer) >= 5:
                        try:
                            start_index = self.buffer.index(0x05)
                            if len(self.buffer) - start_index >= 5:
                                frame = self.buffer[start_index:start_index + 5]
                                self.buffer = self.buffer[start_index + 5:]
                                hex_data = ' '.join(f'{b:02X}' for b in frame)
                                print(f"[找到完整帧] {hex_data}")  # 添加方括号
                                yaw_bytes = frame[1:3]
                                preload_bytes = frame[3:5]

                                # 解析yaw和preload
                                yaw = struct.unpack('<H', yaw_bytes)[0]
                                preload = struct.unpack('<h', preload_bytes)[0]

                                # 数据有效性检查：确保yaw和preload值在合理范围内
                                if yaw < 0 or yaw > 65535:
                                    print(f"[警告] 无效的yaw值: {yaw}")  # 添加方括号
                                    continue  # 跳过无效的数据
                                if preload < -32768 or preload > 32767:
                                    print(f"[警告] 无效的preload值: {preload}")  # 添加方括号
                                    continue  # 跳过无效的数据

                                print(f"[解析后] yaw: {yaw}, preload: {preload}")  # 添加方括号
                                self.last_yaw = yaw  # 保存最后的有效值
                                self.last_preload = preload  # 保存最后的有效值
                                self.data_received.emit(float(yaw), float(preload))
                                return {'yaw': yaw, 'preload': preload}
                            else:
                                time.sleep(0.01)  # 数据不足，继续等待
                                continue
                        except ValueError:
                            print(f"[警告] 未找到帧头，移除第一个字节: {self.buffer.hex()}")  # 添加方括号
                            self.buffer.pop(0)  # 移除错误字节继续尝试
                            continue
                    else:
                        time.sleep(0.01)  # 数据不足，继续等待
                        continue
            except serial.SerialException as e:
                print(f"读取串口数据出错: {e}")
                self.connect_serial()  # 连接出错时尝试重新连接
            except struct.error as e:
                print(f"解析结构体出错: {e}, 缓冲区: {self.buffer.hex()}")
                self.buffer.clear()  # 清空缓冲区
        else:
            self.connect_serial()

    def start_reading(self):
        """启动数据读取线程"""
        while self.running:
            self.read_frame()

    def stop(self):
        """停止串口读取并关闭连接"""
        self.running = False
        if self.serial and self.serial.is_open:
            self.serial.close()
            print(f"串口 {self.port} 已关闭")
            
    def send_combined_frame(self, status, green_status, offset):
        """发送合并帧"""
        if self.serial is None or not self.serial.is_open:
            print("串口未打开，发送失败")
            return False
            
        try:
            frame = bytearray()
            frame.append(0xA5)  # 帧头
            frame.append(status & 0xFF)  # 状态
            frame.append(green_status & 0xFF)  # 绿灯状态
            frame.append((offset >> 8) & 0xFF)  # 偏移高字节
            frame.append(offset & 0xFF)  # 偏移低字节
            
            self.serial.write(frame)
            return True
        except Exception as e:
            print(f"发送数据出错: {e}")
            return False
            
    def close(self):
        """关闭串口连接"""
        self.stop()

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
        
        # 初始化变量
        self.camera = None
        self.camera_list = None
        self.cam_handle = None
        self.opencv_camera = None
        self.is_running = True
        self.device_manager = None
        self.serial_comm = None  # 串口通信对象
        self.prev_frame_time = 0
        self.status_frame_counter = 0
        self.received_angle = 0  # 从串口接收的角度值
        self.received_count = 0  # 从串口接收的圈数值
        self.last_serial_update_time = time.time()

        # 初始化UI
        self.initUI()
        
        # 设置摄像头
        self.setup_camera()
        
        # 设置串口
        self.setup_serial()
        
        # 创建帧率计算定时器
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_camera_frame)
        self.fps_timer.start(30)  # 30ms刷新一次，约为33帧每秒
        
        # 创建串口重新连接定时器
        self.reconnect_timer = QTimer()
        self.reconnect_timer.timeout.connect(self.refresh_connection)
        self.reconnect_timer.start(5000)  # 每5秒尝试重新连接一次

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

    def setup_serial(self):
        """初始化串口通信，使用多线程处理"""
        try:
            # 优先尝试连接主串口
            primary_port = '/dev/my_stm32'
            backup_ports = ['/dev/ttyACM0', '/dev/ttyACM1']
            
            # 创建串口接收对象，使用SerialReceiver类
            self.serial_comm = SerialReceiver(port=primary_port, baudrate=115200)
            
            # 连接信号与槽
            self.serial_comm.data_received.connect(self.update_serial_data_ui)
            
            # 启动线程
            self.serial_comm.start()
            
            # 添加状态帧计数器
            self.status_frame_counter = 0
            print(f"串口接收线程已启动，使用端口：{primary_port}")
            
        except Exception as e:
            print(f"主串口初始化失败: {str(e)}")
            
            # 如果主串口失败，尝试备用端口
            for port in backup_ports:
                try:
                    self.serial_comm = SerialReceiver(port=port, baudrate=115200)
                    self.serial_comm.data_received.connect(self.update_serial_data_ui)
                    
                    # 启动线程
                    self.serial_comm.start()
                    
                    # 添加状态帧计数器
                    self.status_frame_counter = 0
                    print(f"使用备用串口初始化成功，使用端口：{port}")
                    break
                except Exception as backup_err:
                    print(f"备用串口 {port} 初始化失败: {str(backup_err)}")
            else:
                # 所有端口都尝试失败
                print("所有串口尝试都失败，无法初始化串口通信")
                self.serial_comm = None

    def update_connection_status(self, connected):
        """更新串口连接状态 - 仅保留兼容性，SerialReceiver不使用"""
        pass

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

    def update_serial_data_ui(self, angle, count):
        """槽函数，用于更新 UI 上显示的串口数据"""
        # 保存旧值以便检查是否有变化
        old_angle = self.received_angle
        old_count = self.received_count
        
        # 更新值
        self.received_angle = angle
        self.received_count = count
        self.last_serial_update_time = time.time()
        
        # 检查是否有变化，如果有则打印更新信息并进行可视化反馈
        has_changes = False
        if old_angle != self.received_angle:
            print(f"角度数据更新: {old_angle} -> {self.received_angle}")
            has_changes = True
            
        if old_count != self.received_count:
            print(f"圈数数据更新: {old_count} -> {self.received_count}")
            has_changes = True
            
        # 更新UI标签
        self.update_ui_labels(has_changes)

    def update_ui_labels(self, highlight=False):
        """
        更新 UI 标签
        :param highlight: 是否高亮显示（数据刚刚变化时）
        """
        # 如果需要高亮，应用特殊样式
        if highlight:
            # 高亮样式 - 使用背景色突出显示
            self.yaw_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue; background-color: #FFFF99;")
            self.preload_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green; background-color: #FFFF99;")
            
            # 创建一个定时器，在0.5秒后恢复正常样式
            QTimer.singleShot(500, self.reset_label_style)
        
        # 更新标签文本
        self.yaw_label.setText(f"Received Angle: {int(self.received_angle)}")
        self.preload_label.setText(f"Received Count: {int(self.received_count)}")
        
    def reset_label_style(self):
        """恢复标签的默认样式"""
        self.yaw_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
        self.preload_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")

    def update_camera_frame(self):
        """更新摄像头帧并显示"""
        # 计算帧率
        curr_frame_time = time.time()
        fps = 1 / (curr_frame_time - self.prev_frame_time) if hasattr(self, 'prev_frame_time') and self.prev_frame_time != 0 else 0
        self.prev_frame_time = curr_frame_time
        
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
        
        # 如果在最近1秒内有更新，则用绿色显示；1-2秒为黄色；超过2秒则用红色显示
        if time_since_update < 1.0:
            serial_color = (0, 255, 0)  # 绿色，表示数据很新鲜
        elif time_since_update < 2.0:
            serial_color = (0, 255, 255)  # 黄色，表示数据稍旧
        else:
            serial_color = (0, 0, 255)  # 红色，表示数据已经过时
        
        # 显示角度数据，使用接收到的最新值
        cv2.putText(result, f"Received Angle: {int(self.received_angle)}", 
                    (text_start_x, serial_data_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, serial_color, 2)
                    
        # 显示圈数数据，使用接收到的最新值
        cv2.putText(result, f"Received Count: {int(self.received_count)}", 
                    (text_start_x, serial_data_y + line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, serial_color, 2)
                    
        # 显示上次更新时间
        update_time_text = f"Last update: {time_since_update:.1f}s ago"
        cv2.putText(result, update_time_text, 
                    (text_start_x, serial_data_y + line_spacing*2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, serial_color, 2)
            
        # 修改串口通信部分
        if hasattr(self, 'serial_comm') and self.serial_comm is not None and hasattr(self.serial_comm, 'serial') and self.serial_comm.serial is not None and self.serial_comm.serial.is_open:
            try:
                # 将布尔值明确转换为整数
                # 只有当检测到绿灯时，status_int和green_status_int才为1
                status_int = 1 if green_detected and contour_info else 0
                green_status_int = 1 if green_detected and contour_info else 0
                
                # 发送合并帧 - 由于SerialReceiver没有send_combined_frame方法，我们需要直接操作其串口
                frame = bytearray()
                frame.append(0xA5)  # 帧头
                frame.append(status_int & 0xFF)  # 状态
                frame.append(green_status_int & 0xFF)  # 绿灯状态
                frame.append((offset_int >> 8) & 0xFF)  # 偏移高字节
                frame.append(offset_int & 0xFF)  # 偏移低字节
                
                self.serial_comm.serial.write(frame)
                
                # 显示发送信息（仅在控制台）
                if hasattr(self, 'status_frame_counter'):
                    self.status_frame_counter += 1
                    if self.status_frame_counter % 30 == 0:  # 每隔30帧显示一次
                        status_text = "检测到目标" if status_int == 1 else "未检测到目标"
                        green_status_text = "检测到绿灯" if green_status_int == 1 else "未检测到绿灯"
                        
                        print(f"发送合并帧：A5 {status_int:02X} {green_status_int:02X} {(offset_int >> 8) & 0xFF:02X} {offset_int & 0xFF:02X}")
                        print(f"  - 状态: {status_text}")
                        print(f"  - 绿灯: {green_status_text}")
                        print(f"  - 偏移值: {offset_int:d}")
                
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
        if hasattr(self, 'fps_timer'):
            self.fps_timer.stop()
        
        # 关闭相机
        if hasattr(self, 'camera') and self.camera is not None:
            print("关闭相机...")
            self.camera.MV_CC_StopGrabbing()
            self.camera.MV_CC_CloseDevice()
            self.camera.MV_CC_DestroyHandle()
        
        # 停止屏幕录制
        print("停止屏幕录制...")
        stop_screen_recording()
        
        # 停止串口通信线程
        if hasattr(self, 'serial_comm') and self.serial_comm is not None:
            print("停止串口接收线程...")
            self.serial_comm.stop()
            self.serial_comm.wait()  # 等待线程结束
            print("串口接收线程已停止")
        
        print("应用已关闭")
        event.accept()

    def initUI(self):
        """初始化UI，需要创建用于显示 yaw 和 preload 的 QLabel"""
        self.setWindowTitle('相机检测')
        self.setGeometry(100, 100, 1200, 700) # 适当增加高度

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.camera_label)

        self.mask_label = QLabel()
        self.mask_label.setMinimumSize(640, 480)
        right_layout.addWidget(self.mask_label)

        # 创建数据显示区域
        data_label_layout = QHBoxLayout()
        
        # 创建用于显示 yaw 和 preload 的标签，设置样式使其更明显
        self.yaw_label = QLabel("Received Angle: 0")
        self.yaw_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
        self.yaw_label.setMinimumWidth(200)
        
        self.preload_label = QLabel("Received Count: 0")
        self.preload_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        self.preload_label.setMinimumWidth(200)
        
        # 创建连接状态标签
        self.connection_status_label = QLabel("连接状态: 未知")
        self.connection_status_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # 创建刷新按钮
        self.refresh_button = QPushButton("刷新连接")
        self.refresh_button.clicked.connect(self.refresh_connection)
        
        data_label_layout.addWidget(self.yaw_label)
        data_label_layout.addWidget(self.preload_label)
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.connection_status_label)
        control_layout.addWidget(self.refresh_button)
        
        left_layout.addLayout(data_label_layout)
        left_layout.addLayout(control_layout)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        main_widget.setLayout(layout)

    def refresh_connection(self):
        """刷新串口连接"""
        if hasattr(self, 'serial_comm') and self.serial_comm is not None:
            print("尝试重新连接串口...")
            self.serial_comm.connect_serial()
            
            # 更新连接状态标签
            connection_status = "已连接" if (self.serial_comm.serial and self.serial_comm.serial.is_open) else "断开"
            self.connection_status_label.setText(f"连接状态: {connection_status}")
            
            # 更新最后连接时间
            if self.serial_comm.serial and self.serial_comm.serial.is_open:
                self.last_serial_update_time = time.time()

def main():
    # 启动屏幕录制
    start_screen_recording(format="webm", capture_method="xcb")  # 使用WebM格式和XCB捕获方法，确保兼容性
    
    # 注册程序退出时的回调，确保录制停止
    atexit.register(stop_screen_recording)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()