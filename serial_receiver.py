import serial
import struct
import time
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import QApplication

class SerialReceiver(QThread):
    data_received = pyqtSignal(float, float)  # 定义信号，传递 yaw 和 preload

    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.buffer = bytearray()
        self.last_yaw = 0
        self.last_preload = 0
        self.serial = None
        self.running = True
        self.connect_serial()

    def connect_serial(self):
        """连接串口，确保连接成功"""
        try:
            if self.serial is None or not self.serial.is_open:
                self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
                print(f"串口 {self.port} 连接成功")
        except serial.SerialException as e:
            print(f"无法连接串口 {self.port}: {e}")
            self.serial = None

    def read_frame(self):
        """读取并解析数据帧"""
        if self.serial and self.serial.is_open:
            try:
                while self.running:
                    if self.serial.in_waiting > 0:
                        new_data = self.serial.read(self.serial.in_waiting)
                        self.buffer.extend(new_data)
                        print(f"[接收到原始字节流] {new_data.hex()}")  # 打印接收到的原始字节流

                    if len(self.buffer) >= 5:
                        try:
                            start_index = self.buffer.index(0x05)
                            if len(self.buffer) - start_index >= 5:
                                frame = self.buffer[start_index:start_index + 5]
                                self.buffer = self.buffer[start_index + 5:]
                                hex_data = ' '.join(f'{b:02X}' for b in frame)
                                print(f"[找到完整帧] {hex_data}")
                                yaw_bytes = frame[1:3]
                                preload_bytes = frame[3:5]

                                # 解析yaw和preload
                                yaw = struct.unpack('<H', yaw_bytes)[0]
                                preload = struct.unpack('<h', preload_bytes)[0]

                                # 数据有效性检查：确保yaw和preload值在合理范围内
                                if yaw < 0 or yaw > 65535:
                                    print(f"[警告] 无效的yaw值: {yaw}")
                                    continue  # 跳过无效的数据
                                if preload < -32768 or preload > 32767:
                                    print(f"[警告] 无效的preload值: {preload}")
                                    continue  # 跳过无效的数据

                                print(f"[解析后] yaw: {yaw}, preload: {preload}")
                                self.last_yaw = yaw
                                self.last_preload = preload
                                self.data_received.emit(float(yaw), float(preload))
                                return {'yaw': yaw, 'preload': preload}
                            else:
                                time.sleep(0.01)  # 数据不足，继续等待
                                continue
                        except ValueError:
                            print(f"[警告] 未找到帧头，移除第一个字节: {self.buffer.hex()}")
                            self.buffer.pop(0)  # 移除错误字节继续尝试
                            continue
                    else:
                        time.sleep(0.01)  # 数据不足，继续等待
                        continue
            except serial.SerialException as e:
                print(f"读取串口数据出错: {e}")
                self.connect_serial()
            except struct.error as e:
                print(f"解析结构体出错: {e}, 缓冲区: {self.buffer.hex()}")
                self.buffer.clear()  # 清空缓冲区
        else:
            self.connect_serial()

    def run(self):
        """串口数据接收线程"""
        while self.running:
            self.read_frame()

    def stop(self):
        """停止接收线程"""
        self.running = False
        if self.serial and self.serial.is_open:
            self.serial.close()
            print(f"串口 {self.port} 已关闭")

def your_slot_function(yaw, preload):
    """信号槽函数，处理接收到的数据"""
    print(f"接收到数据 - yaw: {yaw}, preload: {preload}")

def main():
    try:
        app = QApplication([])  # 启动 Qt 应用
        receiver = SerialReceiver(port='/dev/ttyACM0', baudrate=115200)
        receiver.data_received.connect(your_slot_function)  # 连接信号槽
        receiver.start()  # 启动接收线程
        print("串口接收器已启动，等待数据...")
        app.exec()  # 进入 Qt 的事件循环
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        if 'receiver' in locals():
            receiver.stop()  # 停止接收线程
            receiver.wait()  # 等待线程完全结束

if __name__ == "__main__":
    main()
