import numpy as np
import cv2
from opencv_green_detection import create_weight_matrix
import webbrowser
import os

def generate_weight_matrix_html(save_path='weight_matrix_viz.html'):
    # 创建一个示例帧大小的权重矩阵
    frame = np.zeros((480, 640, 3))  # 假设使用640x480的分辨率
    weight_matrix = create_weight_matrix(frame)
    
    # 生成热力图颜色
    heatmap = cv2.applyColorMap((weight_matrix * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 保存热力图
    cv2.imwrite('heatmap.png', heatmap)
    
    # 创建HTML内容
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>权重矩阵可视化</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f0f0f0;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .visualization {
                position: relative;
                margin-top: 20px;
            }
            .heatmap {
                width: 100%;
                border: 1px solid #ddd;
            }
            .legend {
                margin-top: 10px;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .legend-gradient {
                width: 200px;
                height: 20px;
                background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000);
                margin: 0 10px;
            }
            .legend-text {
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>权重矩阵可视化</h2>
            <div class="visualization">
                <img src="heatmap.png" class="heatmap" alt="Weight Matrix Heatmap">
                <div class="legend">
                    <span class="legend-text">0.2</span>
                    <div class="legend-gradient"></div>
                    <span class="legend-text">1.0</span>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <h3>说明：</h3>
                <ul>
                    <li>红色区域表示权重为1.0（完全检测）</li>
                    <li>蓝色区域表示权重为0.2（弱检测）</li>
                    <li>中间区域（35%-65%）保持最高权重</li>
                    <li>顶部和底部区域权重降低</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    '''
    
    # 保存HTML文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 获取文件的绝对路径
    abs_path = 'file://' + os.path.abspath(save_path)
    
    # 在默认浏览器中打开HTML文件
    webbrowser.open(abs_path)

if __name__ == '__main__':
    generate_weight_matrix_html() 