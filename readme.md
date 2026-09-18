For robot 1 in rt_test
cd ~/rt_test
colcon build --symlink-install
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_control robot_bringup.launch.py
ros2 run robot_control robot_gui_test
ros2 run rqt_graph rqt_graph


 chmod +x ~/Downloads/arduino-ide_2.3.10_Linux_64bit.AppImage
~/Downloads/arduino-ide_2.3.10_Linux_64bit.AppImage --no-sandbox







For intel realsence 
iit@iit-HP-All-in-One-Desktop-27-cr1xxx:~$ ~/librealsense/build/Release/realsense-viewer




Python                  3.12.3
OpenCV                  4.6.0
pyrealsense2            2.58.3
D456                    detected
Firmware                5.17.3.10
RGB                     1280×720
Depth                   1280×720
Depth scale             0.001 m/unit
Depth camera
fx=fy≈657.952 
cx≈641.238,cy≈362.136
RGB camera
Fx≈644.485, fy≈643.873
Cx≈645.525, cy≈370.698

PHASE 2 — REALSENSE PYTHON PERCEPTION

Step 1: Basic RGB + Depth Acquisition
Status: COMPLETED

Verified:
- Python 3.12.3
- OpenCV 4.6.0
- pyrealsense2 2.58.3
- RealSense D456 detected
- Firmware 5.17.3.10
- RGB stream: 1280 × 720 @ 30 FPS
- Depth stream: 1280 × 720 @ 30 FPS
- Depth scale: 0.001 m/unit

Step 2: RGB-Depth Alignment
Status: WORKING

Verified:
- RGB frame acquisition
- Depth frame acquisition
- Depth aligned to RGB stream
- RGB pixel selection
- Depth retrieval at selected pixel
- Pixel-to-3D deprojection

Step 3: Robust 3D Point Measurement
Status: NEXT

Goal:
Selected RGB pixel
→ neighborhood depth sampling
→ remove invalid depth values
→ median depth
→ deproject to (Xc, Yc, Zc)

Step 4: Camera-to-Robot Calibration
Status: AFTER ROBUST 3D MEASUREMENT

Goal:
Camera-frame coordinates (Xc,Yc,Zc)
→ camera-to-robot transformation ^B T_C
→ robot-base coordinates (Xb,Yb,Zb)


TEST A — CURRENT CONFIGURATION
Quantity measured: Zc only

+----------+--------------+--------------+-------------+-------------+
| Test     | Actual Z     | Mean Zc      | Abs. Error  | Trial Range|
+----------+--------------+--------------+-------------+-------------+
| M1       | 226–227 cm   | 224.73 cm    | 1.27–2.27 cm| 1.7 cm     |
| M2       | 142 cm       | 139.97 cm    | 2.03 cm     | 2.4 cm     |
| M3       | 58 cm        | 55.46 cm     | 2.54 cm     | 0.1 cm     |
+----------+--------------+--------------+-------------+-------------+
OBSERVATION:

The D456 shows good temporal repeatability on the flat box surface,
especially at short range, but there is a systematic negative depth
offset of approximately 1–3 cm compared with the physical measurement.

The repeatability and absolute accuracy are different quantities.

