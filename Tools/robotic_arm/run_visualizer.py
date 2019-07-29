import robotic_visualizer
import time
import threading

view = robotic_visualizer.RoboticVisualizer()
view.read_robotic_arm_pose_list("testing_pose.testingjson")
#     view.update_visualizer()
view.run()

view.read_robotic_arm_pose_list("")
