#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import signal
import time
from datetime import datetime

class SystemRecorder:
    """系统录屏器，使用ffmpeg命令行工具进行屏幕录制"""
    
    def __init__(self, output_folder=None, format="webm", capture_method="xcb"):
        """初始化录屏器
        
        Args:
            output_folder: 视频保存的文件夹路径，默认为DartSystem/video
            format: 视频格式，可选"mp4"、"avi"、"mkv"或"webm"
        """
        self.capture_method = capture_method.lower()  # 捕获方法
        # 设置输出文件夹
        if output_folder is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_folder = os.path.join(script_dir, "video")
            
        self.output_folder = output_folder
        self.recording = False
        self.process = None
        self.output_path = None
        self.format = format.lower()  # 视频格式
        self.capture_method = capture_method.lower()  # 捕获方法
        
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"创建视频保存目录: {output_folder}")
        else:
            print(f"视频将保存到: {output_folder}")
    
    def start_recording(self):
        """开始录制屏幕"""
        if self.recording:
            print("已经在录制中")
            return
        
        try:
            # 检查ffmpeg是否已安装
            try:
                subprocess.run(["which", "ffmpeg"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError:
                print("错误: 未安装ffmpeg。请使用命令安装: sudo apt-get install ffmpeg")
                return
            
            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 根据选择的格式设置文件扩展名
            extension = self.format
            if extension not in ["mp4", "avi", "mkv", "webm"]:
                extension = "webm"  # 默认使用webm
                
            self.output_path = os.path.join(self.output_folder, f"screen_recording_{timestamp}.{extension}")
            print(f"屏幕录制将保存到: {self.output_path}")
            
            # 根据捕获方法使用不同的命令
            if self.capture_method == "fb":
                # 使用framebuffer捕获方法，尝试解决黑屏问题
                print("使用framebuffer捕获方法，尝试解决黑屏问题")
                cmd = [
                    "ffmpeg",
                    "-f", "fbdev",            # 使用framebuffer设备
                    "-framerate", "10",        # 设置帧率
                    "-i", "/dev/fb0",          # framebuffer设备路径
                    "-pix_fmt", "yuv420p",     # 输出像素格式
                    "-y",                      # 覆盖已有文件
                    self.output_path
                ]
            elif self.capture_method == "xcb":
                print("使用XCB捕获方法，解决黑屏问题")
                cmd = [
                    "ffmpeg",
                    "-f", "x11grab",           # 仍使用x11grab，但加入XFixes扩展支持
                    "-draw_mouse", "1",        # 启用鼠标绘制
                    "-s", self._get_screen_resolution(),  # 设置分辨率
                    "-i", ":0.0+0,0",          # 捕获主显示器，指定偏移
                    "-vsync", "1",             # 视频同步
                    "-r", "10",                # 帧率10fps
                    "-c:v", "libvpx",          # 使用VP8编码
                    "-b:v", "2000k",           # 增加比特率  
                    "-pix_fmt", "yuv420p",     # 使用标准像素格式
                    "-deadline", "good",       # 使用更好的质量设置
                    "-threads", "4",           # 增加线程数
                    "-y",                      # 覆盖已有文件
                    self.output_path
                ]
            else:
                print("使用X11捕获方法")
                cmd = [
                    "ffmpeg",
                    "-f", "x11grab",           # 使用x11grab捕获X11显示
                    "-draw_mouse", "1",        # 启用鼠标绘制
                    "-s", self._get_screen_resolution(),  # 设置分辨率
                    "-i", ":0.0+0,0",          # 捕获主显示器，指定偏移
                    "-r", "10",                # 帧率10fps
                    "-c:v", "libvpx",          # 使用VP8编码
                    "-b:v", "2000k",           # 增加比特率
                    "-pix_fmt", "yuv420p",     # 使用标准像素格式
                    "-cpu-used", "0",          # 优化质量而非速度
                    "-auto-alt-ref", "1",      # 启用替代参考帧
                    "-deadline", "good",       # 使用更好的质量设置
                    "-threads", "4",           # 增加线程数
                    "-y",                      # 覆盖已有文件
                    self.output_path
                ]
            
            # 增加调试输出
            print(f"屏幕录制命令: {' '.join(cmd)}")
            
            # 启动录制进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.recording = True
            print(f"开始录制屏幕 (进程ID: {self.process.pid})")
        
        except Exception as e:
            print(f"启动录制失败: {e}")
            self.recording = False
            self.process = None
    
    def stop_recording(self):
        """停止录制屏幕"""
        if not self.recording:
            print("没有正在进行的录制")
            return
        
        try:
            print("正在停止录制...")
            
            if self.process:
                # 向进程发送中断信号
                self.process.send_signal(signal.SIGINT)
                
                # 等待进程结束
                try:
                    print("等待ffmpeg完成视频文件...")
                    # 增加等待时间以确保ffmpeg能完成文件写入
                    self.process.wait(timeout=10)
                    print(f"录制已停止，视频已保存到: {self.output_path}")
                    print(f"绝对路径: {os.path.abspath(self.output_path)}")
                    
                    # 验证生成的文件
                    self._verify_video_file()
                except subprocess.TimeoutExpired:
                    print("警告: ffmpeg没有及时响应，尝试使用SIGTERM信号")
                    # 先尝试SIGTERM，这是更温和的终止方式
                    self.process.send_signal(signal.SIGTERM)
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print("警告: ffmpeg仍未响应，强制终止")
                        self.process.kill()
                        self.process.wait()
            
            # 重置状态
            self.recording = False
            self.process = None
            
        except Exception as e:
            print(f"停止录制时出错: {e}")
    
    def _verify_video_file(self):
        """验证生成的视频文件是否有效"""
        if not os.path.exists(self.output_path):
            print(f"错误: 视频文件不存在: {self.output_path}")
            return False
        
        # 检查文件大小
        file_size = os.path.getsize(self.output_path)
        if file_size < 1000:  # 小于1KB的文件可能有问题
            print(f"警告: 视频文件过小 ({file_size} 字节)，可能无效")
            return False
        
        # 使用ffprobe验证文件
        try:
            cmd = ["ffprobe", "-v", "error", self.output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"警告: 视频文件可能损坏: {result.stderr}")
                return False
            print("视频文件验证通过")
            return True
        except Exception as e:
            print(f"验证视频文件时出错: {e}")
            return False
    
    def _get_screen_resolution(self):
        """获取屏幕分辨率"""
        try:
            # 使用xdpyinfo命令获取屏幕分辨率
            result = subprocess.run(
                "xdpyinfo | grep dimensions | awk '{print $2}'",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            
            # 默认分辨率
            return "1920x1080"
        except:
            return "1920x1080"

# 全局录制器实例
recorder = None

def start_system_recording(format="webm", capture_method="xcb"):
    """启动系统录制
    
    Args:
        format: 视频格式，可选"mp4"、"avi"、"mkv"或"webm"，默认使用webm（最兼容）
        capture_method: 屏幕捕获方法，可选"x11"、"xcb"或"fb"，默认使用"xcb"解决黑屏问题
    """
    global recorder
    print(f"正在启动系统屏幕录制(格式: {format}, 捕获方法: {capture_method})...")
    try:
        recorder = SystemRecorder(format=format, capture_method=capture_method)
        recorder.start_recording()
        print("系统屏幕录制已成功启动!")
    except Exception as e:
        print(f"启动系统屏幕录制失败: {e}")

def stop_system_recording():
    """停止系统录制"""
    global recorder
    print("正在停止系统屏幕录制...")
    try:
        if recorder:
            recorder.stop_recording()
            print("系统屏幕录制已成功停止!")
        else:
            print("没有正在进行的录制")
    except Exception as e:
        print(f"停止系统屏幕录制失败: {e}")

# 在命令行直接运行此脚本时，提供简单的录制功能
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="系统屏幕录制工具")
    parser.add_argument("action", choices=["start", "stop", "record"], 
                        help="操作: start(开始录制), stop(停止录制), record(录制指定秒数)")
    parser.add_argument("-d", "--duration", type=int, default=10,
                        help="录制时长(秒)，默认10秒，仅当action=record时有效")
    args = parser.parse_args()
    
    if args.action == "start":
        start_system_recording()
    elif args.action == "stop":
        stop_system_recording()
    elif args.action == "record":
        print(f"开始录制，将持续{args.duration}秒...")
        start_system_recording()
        time.sleep(args.duration)
        stop_system_recording()
        print("录制完成！") 