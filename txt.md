**目标:**

修改一个名为 `main.py` 的 Python 脚本，使其从一个名为 `config.yaml` 的配置文件中加载相机的曝光（exposure）和增益（gain）参数，而不是在代码中硬编码这些值。之前的自动化代码编辑尝试未能成功应用更改。

**需要创建的文件 (`config.yaml`):**

请创建一个名为 `config.yaml` 的文件，其内容如下：

```yaml
camera:
  exposure: 16000.0 # 曝光时间 (us)
  gain: 15.9     # 增益值 (dB)
serial:
  primary_port: '/dev/my_stm32'
  baudrate: 115200
  backup_ports:
    - '/dev/ttyACM0'
    - '/dev/ttyACM1'
```

**需要对 `main.py` 进行的修改:**

1.  **导入 `yaml` 库:** 在脚本的开头添加 `import yaml`。
2.  **修改 `main` 函数:**
    *   在函数开始处，添加代码以打开并读取 `config.yaml` 文件，使用 `yaml.safe_load()` 解析内容。
    *   包含错误处理逻辑，以应对文件未找到 (`FileNotFoundError`) 或 YAML 解析错误 (`yaml.YAMLError`) 的情况。如果加载失败，打印错误信息并退出程序 (`sys.exit(1)`)。
    *   检查加载的 `config` 字典是否有效（例如，非 `None` 且包含 `'camera'` 键）。
    *   在创建 `MainWindow` 实例时，将加载到的相机配置 (`config['camera']`) 作为参数传递给构造函数。例如：`window = MainWindow(camera_config=config['camera'])`。
3.  **修改 `MainWindow` 类的 `__init__` 方法:**
    *   将 `__init__` 方法的签名从 `def __init__(self):` 修改为 `def __init__(self, camera_config):` 以接收配置字典。
    *   在方法内部（例如，在 `super().__init__()` 之后），添加一行代码 `self.camera_config = camera_config` 来存储这个配置。
    *   修改调用 `self.setup_camera()` 的地方，将存储的配置传递给它：`self.setup_camera(self.camera_config)`。
4.  **修改 `MainWindow` 类的 `setup_camera` 方法:**
    *   将 `setup_camera` 方法的签名从 `def setup_camera(self):` 修改为 `def setup_camera(self, camera_config):` 以接收配置字典。
    *   找到调用 `set_Value` 函数来设置 "ExposureTime" 和 "Gain" 的代码行。
    *   修改这些行，使其从传入的 `camera_config` 字典中获取值，而不是使用硬编码的数字。例如：
        *   `set_Value(self.camera, "float_value", "ExposureTime", camera_config['exposure'])`
        *   `set_Value(self.camera, "float_value", "Gain", camera_config['gain'])`

**请求:**

请提供修改后的完整 `main.py` 文件内容。