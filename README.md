1. **进入项目目录**:

```bash
   cd /racing/guo_dev_ws
```

2. **进入编译目录**:

   ```bash
   cd build
   ```
3. **若修改文件，运行 CMake 生成 Makefile**:

   ```bash
   cmake ..
   ```
4. **执行编译**:

   ```bash
   make
   ```

编译成功后，可执行文件会生成在 `build` 目录下。

### 运行程序

 main.cpp编译后会生成一个名为 `SmartCar` 的主程序。

现在先在track_line.cpp中填充代码

```bash
# 在 build 目录下执行
# main.cpp:
./SmartCar
# track_line.cpp:
./track_line
```

**注意**: `12.2/src/main.cpp` 中的代码指定了 YOLO 模型需要存放在 `/home/pi/model/` 目录下，请确保已将模型文件（`.mnn` 文件）放置在正确的位置。

**对于 `dev_ws` 项目**:
编译后会生成多个可执行文件，如 `motor_test`, `track_line`, `avoid` 等。可以根据需要运行相应的程序来测试特定功能。

```bash
# 在 build 目录下执行，例如运行电机测试程序
./motor_test
```

如果要display在电脑上：

1.打开Xlaunch,在主目录下输入 export DISPLAY=ip+:0.0
比如szc的IP是192.168.137.1 输入export DISPLAY=192.168.137.1:0.0
2.运行.py文件
