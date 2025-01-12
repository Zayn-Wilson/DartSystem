import threading
import time
import cv2
import numpy as np
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from ctypes import *

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

def hik_camera_get():
    global camera_image
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
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("创建句柄失败! ret[0x%x]" % ret)
        return

    # 打开设备
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("打开设备失败! ret[0x%x]" % ret)
        return

    # 获取并打印当前参数
    print("当前参数:")
    print("曝光时间:", get_Value(cam, "float_value", "ExposureTime"))
    print("增益:", get_Value(cam, "float_value", "Gain"))
    print("触发模式:", get_Value(cam, "enum_value", "TriggerMode"))
    print("帧率:", get_Value(cam, "float_value", "AcquisitionFrameRate"))

    # 设置相机参数
    set_Value(cam, "float_value", "ExposureTime", 16000)  # 曝光时间
    set_Value(cam, "float_value", "Gain", 15.9)  # 增益值

    # 开始取流
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("开始取流失败! ret[0x%x]" % ret)
        return

    # 获取数据包大小
    stParam = MVCC_INTVALUE_EX()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
    ret = cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
    if ret != 0:
        print("获取数据包大小失败! ret[0x%x]" % ret)
        return

    data_size = stParam.nCurValue
    pData = (c_ubyte * data_size)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, data_size, stFrameInfo, 1000)
        if ret == 0:
            # 打印像素格式
            print(f"Pixel Format: {stFrameInfo.enPixelType}")
            
            # 转换图像格式
            data = np.frombuffer(pData, dtype=np.uint8)
            
            # 根据像素格式处理图像
            frame = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            
            if stFrameInfo.enPixelType == 17301505:  # Mono8
                camera_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif stFrameInfo.enPixelType == 17301513:  # BayerGB8
                # 尝试不同的Bayer模式
                camera_image = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)  # 使用BG2BGR而不是GB2BGR
            elif stFrameInfo.enPixelType == 17301514:  # BayerRG8
                camera_image = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            elif stFrameInfo.enPixelType == 17301515:  # BayerGR8
                camera_image = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            elif stFrameInfo.enPixelType == 17301516:  # BayerBG8
                camera_image = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            else:
                print(f"未知的像素格式，尝试其他转换方式")
                try:
                    camera_image = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                except:
                    try:
                        camera_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    except:
                        print("无法转换图像格式")
                        continue
        else:
            print("获取一帧图像失败: ret[0x%x]" % ret)

        if not camera_running:
            break

    # 停止取流
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("停止取流失败! ret[0x%x]" % ret)

    # 关闭设备
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("关闭设备失败! ret[0x%x]" % ret)

    # 销毁句柄
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("销毁句柄失败! ret[0x%x]" % ret)

class CameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Camera View')
        self.setGeometry(100, 100, 1400, 1000)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # 创建图像显示标签
        self.image_label = QLabel()
        self.image_label.setMinimumSize(1280, 720)
        layout.addWidget(self.image_label)

        # 创建定时器更新图像
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms更新一次

    def update_frame(self):
        if camera_image is not None:
            try:
                # 转换图像格式并显示
                h, w, ch = camera_image.shape
                bytes_per_line = ch * w
                
                # 使用BGR格式显示
                qt_image = QImage(camera_image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
                self.image_label.setPixmap(scaled_pixmap)
            except Exception as e:
                print(f"显示图像时出错: {str(e)}")

    def closeEvent(self, event):
        global camera_running
        camera_running = False
        event.accept()

if __name__ == '__main__':
    camera_running = True
    camera_image = None
    
    # 启动相机线程
    camera_thread = threading.Thread(target=hik_camera_get)
    camera_thread.start()

    # 创建Qt应用
    app = QApplication(sys.argv)
    window = CameraWindow()
    window.show()
    
    # 运行应用
    app.exec()
    
    # 等待相机线程结束
    camera_thread.join()
