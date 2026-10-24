import os, re
d = 'src/robot_control/robot_control/nodes'
for f in os.listdir(d):
    if not f.endswith('.py'): continue
    path = os.path.join(d, f)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Strip leading slash in topic names for subscriptions and publishers
    # Looks for '/something' inside quotes
    content = re.sub(r"(['\"])/([a-zA-Z_]+)(['\"])", r"\1\2\3", content)
    
    # Also update robot_kinematics import in trajectory_node
    if f == 'trajectory_node.py':
        content = content.replace("from robot_control.robot_kinematics import", "from robot_control.kinematics.robot_kinematics import KinematicsConfig\n# removed old import")
        content = content.replace("from robot_control.trajectory_math import", "from robot_control.trajectory.trajectory_math import")
        # Trajectory node needs to instantiate config and pass it, but wait, the easiest way for trajectory_node is to just 
        # initialize config and pass it to pick_and_place, but wait, pick_and_place is not called directly here. 
        # Let's just fix the topics here and manually fix trajectory_node next.
        
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)
print('Topics updated!')
