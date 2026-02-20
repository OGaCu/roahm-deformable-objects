from crisp_py.robot import Robot
import time
left_arm = Robot(namespace="")
left_arm.wait_until_ready()
while(True):
    start_time = time.time()
    p = left_arm.end_effector_pose.copy()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(p)
    # time.sleep(1.0)