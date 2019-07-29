import pose_align
import pose_convert
import visualization
import open3d
import robotic_visualizer


trans_list = pose_convert.read_robotic_pose_as_trans("testing_pose.testingjson", exclude_beginning_n=40)


# optimizer = pose_align.PoseGraphOptimizerG2oRobotic()
# optimizer.load_trans_list(trans_list)
# optimizer.optimize()
# optimizer.save_optimised_as_robotic_pose("aligned_pose.testingjson")


robotic_pose_list_original = pose_convert.read_robotic_pose("testing_pose.testingjson")
points_original = pose_convert.robotic_pose_list_to_points(robotic_pose_list_original)

robotic_pose_list = pose_convert.read_robotic_pose("aligned_pose.testingjson")
points = pose_convert.robotic_pose_list_to_points(robotic_pose_list)

# navigator = navigation.NavigationGraph()
# navigator.load_point_list(points, 0.03)
# order = navigator.dfs()
#
#
# # points = pose_convert.adjust_order(points, order)
# # robotic_pose_list = pose_convert.adjust_order(robotic_pose_list, order)
#
pcd_original = visualization.make_point_cloud(points_original, color=[1, 0, 0])
pcd = visualization.make_point_cloud(points)
route = visualization.make_connection_of_pcd_order(pcd)

open3d.draw_geometries([pcd_original])
open3d.draw_geometries([pcd_original, pcd, route])


viewer = robotic_visualizer.RoboticVisualizer()
viewer.read_robotic_arm_pose_list("aligned_pose.testingjson")
viewer.run()