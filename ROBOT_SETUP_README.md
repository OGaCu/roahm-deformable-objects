# Move some robots

## Data Collection Pipeline

There are a lot of terminals to open and commands to run to get this to work

Do the following in order:

**Launch Dual Franka**

Terminal 1:

```
cd move_some_robots/crisp/crisp_controllers_demos/
LEFT_ROBOT_IP=192.168.2.3 RIGHT_ROBOT_IP=192.168.2.2 docker compose up launch_dual_franka
```

**Launch Camera ROS Dockers**

Terminal 2:
```
cd move_some_robots/crisp/crisp_controllers_demos/
docker compose up launch_realsense_camera_right
```

Terminal 3:
```
cd move_some_robots/crisp/crisp_controllers_demos/
docker compose up launch_realsense_camera_wrist
```
Terminal 4:
```
docker exec -it crisp_controllers_demos_realsense_right /bin/bash
source /opt/ros/humble/setup.bash
ros2 topic list # you should see topics in the format '/right/right_wall_camera' or '/wrist/wrist_XXX'
rqt # you should see a camera feed
```

**Start Gripper**

Terminal 5:
```
cd move_some_robots/crisp_env/crisp_py
pixi shell -e humble
python examples/08c.py
```
Use the SPACE bar or attach a clicker and press the bottom right button to toggle the gripper


**Start Recording**
```
cd move_some_robots/crisp_gym
pixi shell -e humble-lerobot
python scripts/record_lerobot_format_leader_follower.py
```
Press **r** to start recording, **p** to pause the recording, **s** to save the recording, and **q** to end the program.

The following parameters are worth changing when you run the python script: `task`, `repo_id`, `fps`, `num_episodes`, `resume`.

```
python scripts/record_lerobot_format_leader_follower.py \
    --task "pour the water" \
    --fps 10 \
    --num-episodes 5 \
    --resume
```

Videos are stored in `~/.cache/huggingface/{repo_id}/videos`. To see a video and check if it works, use vlc to open the video (the video won't load in any other way):

```
vlc ~/.cache/huggingface/lerobot/test/videos/chunk-000/observation.images.{wall or wrist}/episode_{episode_number formatted to 6 digits}.mp4
```

**Note**: Repo id has to match one that's in huggingface.

**ALT: Start teleop without recording**
```
cd move_some_robots/crisp_gym
pixi shell -e humble-lerobot
python examples/03b.py
```


## Using 1 Robot

### Terminal 1:
```
cd move_some_robots/crisp/crisp_controllers_demos/
ROBOT_IP=192.168.2.2 docker compose up launch_franka
```
192.168.2.3 for the left robot \
192.168.2.2 for the right robot

### Terminal 2:
```
cd move_some_robots/crisp_env/crisp_py
pixi shell -e humble
python examples/02b_joint_controls.py
```

### Explanation of the codes
*`crisp_env/crisp_py/examples/02b_joint_controls.py`* \
Will continuously prompt you to enter an input in `a,b` format where a is an int [0,7] representing the joint number and b is a float [-1,1] representing how much to move it by.

## Using 2 Robots

### Terminal 1:

```
LEFT_ROBOT_IP=192.168.2.3 RIGHT_ROBOT_IP=192.168.2.2 docker compose up launch_dual_franka
```

### Terminal 2:

cd move_some_robots/crisp_gym
pixi shell -e humble-lerobot
python examples/03a.py

### Explanation of the codes
*`crisp_gym/examples/03b.py`* \
Will allow teleoperation (hard coded such that right is leader and left is follower). Note: the gripper doesn't work

*`crisp_env/crisp_py/examples/08c.py`* \
This will allow you to control the gripper by publishing to a ros node.
Use SPACE ( ) or PERIOD (.) to toggle the gripper open or closed. (Hint: on a clicker, the present button is period)

## Connect Cameras

### Terminal 1a
```
docker compose up launch_realsense_camera_right
```
### Terminal 1b
```
docker compose up launch_realsense_camera_wrist
```
### Terminal 2
```
docker exec -it crisp_controllers_demos_realsense_right /bin/bash
source /opt/ros/humble/setup.bash
ros2 topic list # you should see topics in the format '/right/right_wall_camera' or '/wrist/wrist_XXX'
rqt # you should see a camera feed
```

## Important notes

### Issue with joint_control.yaml

When running a teleop example in `crisp_gym` there was an error:
`joint_control.yaml` was hard coded to be for the 'right' robot, however I want the left robot to be the follower.
`jc.yaml` is in crisp_py, which is a different folder than crisp_gym (the crisp_py that's in this repo isn't what's used, it's `pip` installed separately).

To fix this, I modified `move_some_robots/crisp_gym/.pixi/envs/humble-lerobot/lib/python3.11/site-packages/crisp_py/config/control/joint_control.yaml` to have each `nullspace.weights.right_fr3_jointX.value` to be `nullspace.weights.left_fr3_jointX.value`. Please note this if you need to make the same modification.


# Running up Pi0

1. Follow steps for setting up 1 robot
2. Run the following commands (they are written where it's assumed you're already in `move_some_robots/`):

```
cd openpi
uv sync
cd third_party/crisp_py
pixi shell -e humble
cd ../..
./examples/run_move_some_robot.sh
```
Before running, make sure that the robot is active, the gripper is plugged in, and the camera is connected to the computer.

To modify the task, modify `openpi/examples/task.txt`. Once you save the file, the task will auto update in the running script.