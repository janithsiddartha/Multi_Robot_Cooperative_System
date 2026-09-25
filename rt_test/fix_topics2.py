import os
import re

d = r'C:\Users\IIT\Desktop\rt_test\src\robot_control\robot_control\nodes'
files = ['trajectory_node.py', 'uart_node.py', 'visualizer_node.py', 'gripper_mux_node.py', 'control_mode_manager_node.py', 'command_mux_node.py']

for f in files:
    path = os.path.join(d, f)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace leading slash in topic strings
    content = re.sub(r"(['\"])/([a-zA-Z_]+)(['\"])", r"\1\2\3", content)
    
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)
        
print("Successfully replaced topics")
