import navigation
import pose_convert
import visualization
import open3d

robotic_pose_list_original = pose_convert.read_robotic_pose("testing_pose.testingjson", exclude_beginning_n=40)
points_original = pose_convert.robotic_pose_list_to_points(robotic_pose_list_original)

robotic_pose_list = pose_convert.read_robotic_pose("interpolated.testingjson")
points = pose_convert.robotic_pose_list_to_points(robotic_pose_list)



# pose_convert.non_outlier_index_list(points_original, 20, 0.1)
# pose_convert.non_outlier_index_list_radius(points_original, 20, 0.005)


navigator = navigation.NavigationGraph()
navigator.load_point_list(points, 0.03)
order = navigator.dfs()
# order = navigator.bfs(2)
#
#
points = pose_convert.adjust_order(points, order)
robotic_pose_list = pose_convert.adjust_order(robotic_pose_list, order)

pose_convert.save_robotic_pose("ordered.testingjson", robotic_pose_list)

pcd_original = visualization.make_point_cloud(points_original, color=[1, 0, 0])
pcd = visualization.make_point_cloud(points)
route = visualization.make_connection_of_pcd_order(pcd)

open3d.draw_geometries([pcd_original, pcd, route])
