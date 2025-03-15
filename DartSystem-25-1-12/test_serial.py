from serial.serialposix import Serial
import time
import sys
import struct
import crcmod.predefined
from threading import Thread

class SerialTest:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        try:
            self.ser = Serial(
                port=port,
                baudrate=baudrate,
                timeout=1
            )
            self.crc16 = crcmod.predefined.Crc('crc-16')
            print(f"成功打开串口 {port}")
            self.running = True
        except Exception as e:
            print(f"打开串口失败: {str(e)}")
            sys.exit(1)

    def calculate_crc(self, data):
        """计算CRC"""
        self.crc16.new()  # 重置CRC计算器
        self.crc16.update(data)
        return self.crc16.digest()

    def build_frame(self, data_value):
        """构建帧：帧头(0xA5) + 数据段 + CRC"""
        frame_header = 0xA5
        if isinstance(data_value, int):
            data_segment = struct.pack('>h', data_value)  # 偏差值用2字节
        else:
            raise ValueError("只能发送整数数据")
        temp_frame = struct.pack('B', frame_header) + data_segment
        crc = self.calculate_crc(temp_frame)
        complete_frame = temp_frame + crc
        return complete_frame

    def send_test(self):
        """发送测试数据"""
        count = 0
        try:
            while self.running:
                # 发送偏差值测试数据
                test_value = count % 200 - 100  # 在 -100 到 99 之间循环
                frame = self.build_frame(test_value)
                self.ser.write(frame)
                print(f"发送帧 (hex): {frame.hex()}, 偏差值: {test_value}")
                count += 1
                time.sleep(0.5)  # 每0.5秒发送一次
        except KeyboardInterrupt:
            print("\n停止发送")

    def receive_test(self):
        """接收数据"""
        try:
            while self.running:
                if self.ser.in_waiting:
                    # 读取帧头
                    header = self.ser.read(1)
                    if not header:
                        continue
                    
                    if header[0] == 0xA5:
                        # 读取数据段（2字节）
                        data = self.ser.read(2)
                        if len(data) == 2:
                            # 读取CRC（2字节）
                            crc = self.ser.read(2)
                            if len(crc) == 2:
                                # 验证CRC
                                frame = header + data
                                calculated_crc = self.calculate_crc(frame)
                                if crc == calculated_crc:
                                    # 解析数据
                                    value = struct.unpack('>h', data)[0]
                                    print(f"接收到有效数据: {value}")
                                else:
                                    print(f"CRC校验失败")
                            else:
                                print("接收CRC失败")
                        else:
                            print("接收数据段失败")
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n停止接收")

    def send_and_receive(self):
        """同时发送和接收数据"""
        send_thread = Thread(target=self.send_test)
        receive_thread = Thread(target=self.receive_test)
        
        send_thread.start()
        receive_thread.start()
        
        try:
            send_thread.join()
            receive_thread.join()
        except KeyboardInterrupt:
            self.running = False
            print("\n停止发送和接收")

    def close(self):
        """关闭串口"""
        self.running = False
        if self.ser.is_open:
            self.ser.close()
            print("串口已关闭")

def main():
    # 创建串口测试实例
    serial_test = SerialTest()

    # 选择模式
    print("\n选择测试模式:")
    print("1: 仅发送数据")
    print("2: 仅接收数据")
    print("3: 同时发送和接收")
    mode = input("请选择模式 (1/2/3): ").strip()

    try:
        if mode == "1":
            print("开始发送数据 (按 Ctrl+C 停止)...")
            serial_test.send_test()
        elif mode == "2":
            print("开始接收数据 (按 Ctrl+C 停止)...")
            serial_test.receive_test()
        elif mode == "3":
            print("开始发送和接收数据 (按 Ctrl+C 停止)...")
            serial_test.send_and_receive()
        else:
            print("无效的模式选择")
    finally:
        serial_test.close()

if __name__ == "__main__":
    main()
