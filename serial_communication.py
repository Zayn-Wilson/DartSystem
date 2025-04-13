from serial.serialposix import Serial
import struct
import crcmod.predefined
import serial
import time

class SerialCommunication:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.serial = serial.Serial(port, baudrate)
        # 初始化接收的数据字段
        self.yaw = 0
        self.preload = 0
        self.last_receive_time = 0
        self.is_data_updated = False

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
        
        # 注释掉发送后立即接收数据的部分
        # self.receive_data()

    # 注释掉接收数据的方法
    """
    def receive_data(self):
        '''
        接收电控发过来的数据
        帧格式：0x05 + yaw(4字节) + preload(4字节)
        返回: 接收成功返回True，否则返回False
        '''
        try:
            # 检查是否有足够的数据可以读取
            if self.serial.in_waiting >= 9:  # 一个字节的帧头 + 4字节yaw + 4字节preload
                # 读取帧头
                header = self.serial.read(1)
                if header[0] == 0x05:  # 检查是否是正确的帧头
                    # 读取yaw值（4字节浮点数）
                    yaw_bytes = self.serial.read(4)
                    # 读取preload值（4字节浮点数）
                    preload_bytes = self.serial.read(4)
                    
                    # 解析数据（假设使用小端格式）
                    try:
                        self.yaw = struct.unpack('<f', yaw_bytes)[0]
                        self.preload = struct.unpack('<f', preload_bytes)[0]
                        self.last_receive_time = time.time()
                        self.is_data_updated = True
                        return True
                    except struct.error as e:
                        print(f"解析数据出错: {e}")
                        return False
                else:
                    # 无效的帧头，丢弃这个字节
                    return False
        except Exception as e:
            print(f"接收数据时出错: {e}")
            return False
        return False

    def check_for_data(self):
        '''
        检查是否有新数据可用，不阻塞
        返回: 如果有新数据则返回True，否则返回False
        '''
        if self.serial.in_waiting > 0:
            return self.receive_data()
        return False
    """

    def close(self):
        if self.serial.is_open:
            self.serial.close()