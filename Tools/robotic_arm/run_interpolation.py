import robotic_surface_interpolation
import robotic_visualizer

surface_interpolator = robotic_surface_interpolation.RoboticSurfaceConstructor()
surface_interpolator.read_robotic_pose_list("testing_pose.testingjson")

surface_interpolator.run_interpolation()
surface_interpolator.save_interpolated_robotic_pose("interpolated.testingjson")

viewer = robotic_visualizer.RoboticVisualizer()
viewer.read_robotic_arm_pose_list("interpolated.testingjson")
viewer.run()

