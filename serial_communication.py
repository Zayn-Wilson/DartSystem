from serial.serialposix import Serial
import struct
import crcmod.predefined
import serial

class SerialCommunication:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.serial = serial.Serial(port, baudrate)

    def send_status_frame(self, status_int):
        """
        发送状态帧
        帧格式：0xA5 + 0x01 + status(0x00/0x01)
        
        参数:
            status_int: 整数，0表示未检测到，1表示检测到
        """
        frame = bytearray([0xA5, 0x01, status_int & 0xFF])
        self.serial.write(frame)

    def send_green_light_frame(self, status_int):
        """
        发送绿灯识别状态帧
        帧格式：0xA5 + 0x02 + status(0x00/0x01)
        
        参数:
            status_int: 整数，0表示未检测到绿灯，1表示检测到绿灯
        """
        frame = bytearray([0xA5, 0x02, status_int & 0xFF])
        self.serial.write(frame)

    def send_offset_frame(self, offset_int):
        """
        发送偏差值帧
        帧格式：0xA5 + 0x03 + offset(2字节)
        
        参数:
            offset_int: 整数，表示偏移值
        """
        frame = bytearray([
            0xA5,                       # 帧头
            0x03,                       # 类型（偏差值帧）
            (offset_int >> 8) & 0xFF,   # 高字节
            offset_int & 0xFF           # 低字节
        ])
        self.serial.write(frame)

    def send_combined_frame(self, status_int, green_light_int, offset_int):
        """
        发送合并帧，包含所有数据
        帧格式：0xA5 + status + green_light + offset_high + offset_low
        
        参数:
            status_int: 整数，0表示未检测到目标，1表示检测到目标
            green_light_int: 整数，0表示未检测到绿灯，1表示检测到绿灯
            offset_int: 整数，表示偏移值
        """
        frame = bytearray([
            0xA5,                       # 帧头
            status_int & 0xFF,          # 状态
            green_light_int & 0xFF,     # 绿灯识别状态
            (offset_int >> 8) & 0xFF,   # 偏移值高字节
            offset_int & 0xFF           # 偏移值低字节
        ])
        self.serial.write(frame)

    def close(self):
        if self.serial.is_open:
            self.serial.close()