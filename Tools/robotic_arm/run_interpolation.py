import robotic_surface_interpolation
import robotic_visualizer
import pose_convert

surface_interpolator = robotic_surface_interpolation.RoboticSurfaceConstructor()

robotic_pose_list = pose_convert.read_robotic_pose("aligned_pose.testingjson")

surface_interpolator.load_robotic_pose_list(robotic_pose_list)

surface_interpolator.run_interpolation()
surface_interpolator.save_interpolated_robotic_pose("interpolated.testingjson")

# viewer = robotic_visualizer.RoboticVisualizer()
# viewer.read_robotic_arm_pose_list("interpolated.testingjson")
# viewer.run()

