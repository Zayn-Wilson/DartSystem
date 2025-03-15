from serial.serialposix import Serial
import struct
import crcmod.predefined
import serial

class SerialCommunication:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.serial = serial.Serial(port, baudrate)

    def send_status_frame(self, detected):
        """
        发送状态帧
        帧格式：0xA5 + 0x01 + status(0x00/0x01)
        """
        status = 0x01 if detected else 0x00
        frame = bytearray([0xA5, 0x01, status])
        self.serial.write(frame)

    def send_offset_frame(self, offset):
        """
        发送偏差值帧
        帧格式：0xA5 + 0x02 + offset(2字节)
        """
        frame = bytearray([
            0xA5,                   # 帧头
            0x02,                   # 类型（偏差值帧）
            (offset >> 8) & 0xFF,   # 高字节
            offset & 0xFF           # 低字节
        ])
        self.serial.write(frame)

    def close(self):
        if self.serial.is_open:
            self.serial.close()