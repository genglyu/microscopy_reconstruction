from open3d import *
import numpy
import sys
sys.path.append("./Utility")
sys.path.append("./CoreModules")
sys.path.append("./Tools")
from tile_info_processing import *
from image_processing import *


def make_tile_frame(trans_matrix, width_by_mm, height_by_mm, color=[0.5, 0.5, 0.5]):
    tile_frame = LineSet()
    lb_rb_rt_lt = [[-width_by_mm / 2, -height_by_mm / 2, 0],
                   [ width_by_mm / 2, -height_by_mm / 2, 0],
                   [ width_by_mm / 2,  height_by_mm / 2, 0],
                   [-width_by_mm / 2,  height_by_mm / 2, 0]
                   ]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    colors = [color, color, color, color]

    tile_frame.points = Vector3dVector(lb_rb_rt_lt)
    tile_frame.lines = Vector2iVector(lines)
    tile_frame.colors = Vector3dVector(colors)
    tile_frame.transform(trans_matrix)

    return tile_frame


def visualize_tile_info_dict_as_point_cloud(tile_info_dict, downsample_factor=1):
    draw_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        img = cv2.imread(tile_info.image_path)
        pcd = load_image_as_planar_point_cloud_open3d(image_bgr=img,
                                                      width_by_mm=tile_info.width_by_mm,
                                                      height_by_mm=tile_info.height_by_mm,
                                                      cv_scale_factor=downsample_factor)
        pcd.transform(tile_info.pose_matrix)
        draw_list.append(pcd)
    draw_geometries(draw_list)


def visualize_tile_info_dict(tile_info_dict, show_point=True, show_edge=True,
                             show_key_frame=True, show_regular_frame=True,
                             show_sensor_frame=False, show_senor_point=False):
    index_list = []
    # geometries would be used for drawing.
    tile_key_frame_list = []
    tile_regular_frame_list = []
    tile_info_point_cloud = PointCloud()
    edge_set = LineSet()
    sensor_point_cloud = PointCloud()
    sensor_edge_set = LineSet()

    # Data used to make the geometries
    tile_positions = []
    tile_colors = []
    tile_normals = []

    sensor_positions = []
    sensor_colors = []
    tile_sensor_pos = []
    sensor_edges = []
    sensor_edge_colors = []

    edges = []
    edge_colors = []

    # make the point cloud of nodes.
    for tile_info_index in tile_info_dict:
        tile_info = tile_info_dict[tile_info_index]
        index_list.append(tile_info_index)

        tile_position = numpy.dot(tile_info.pose_matrix, numpy.asarray([0, 0, 0, 1]).T).T[0:3]
        sensor_position = numpy.dot(tile_info.init_transform_matrix, numpy.asarray([0, 0, 0, 1]).T).T[0:3]

        tile_positions.append(tile_position)
        sensor_positions.append(sensor_position)

        tile_sensor_pos.append(tile_position)
        tile_sensor_pos.append(sensor_position)

        sensor_edges.append([index_list.index(tile_info_index) * 2, index_list.index(tile_info_index) * 2 + 1])
        sensor_edge_colors.append([0.8, 0.8, 0.8])

        if tile_info.is_keyframe:
            tile_colors.append([1, 0, 0])
            sensor_colors.append([0.7, 0.7, 1.0])
            tile_key_frame_list.append(make_tile_frame(trans_matrix=tile_info.pose_matrix,
                                                       width_by_mm=tile_info.width_by_mm,
                                                       height_by_mm=tile_info.height_by_mm,
                                                       color=[1, 0.8, 0.8]))
        else:
            tile_colors.append([0, 0, 0])
            sensor_colors.append([0.7, 0.7, 1.0])
            tile_regular_frame_list.append(make_tile_frame(trans_matrix=tile_info.pose_matrix,
                                                           width_by_mm=tile_info.width_by_mm,
                                                           height_by_mm=tile_info.height_by_mm,
                                                           color=[0.8, 0.8, 0.8]))
        tile_normals.append(
            numpy.dot(tile_info.pose_matrix, numpy.asarray([0, 0, 1, 0]).T).T[0:3]
        )
    tile_info_point_cloud.points = Vector3dVector(tile_positions)
    tile_info_point_cloud.colors = Vector3dVector(tile_colors)
    tile_info_point_cloud.normals = Vector3dVector(tile_normals)
    tile_info_point_cloud.normalize_normals()

    sensor_point_cloud.points = Vector3dVector(sensor_positions)
    sensor_point_cloud.colors = Vector3dVector(sensor_colors)

    sensor_edge_set.points = Vector3dVector(tile_sensor_pos)
    sensor_edge_set.lines = Vector2iVector(sensor_edges)
    sensor_edge_set.colors = Vector3dVector(sensor_edge_colors)

    # make outline frame
    for tile_info_index in tile_info_dict:
        tile_info = tile_info_dict[tile_info_index]
        for odometry_t_id in tile_info.odometry_list:
            edges.append([index_list.index(tile_info_index), index_list.index(odometry_t_id)])
            if index_list.index(tile_info_index) + 1 == index_list.index(odometry_t_id):
                edge_colors.append([1, 0.7, 0.7])
            else:
                edge_colors.append([0.7, 1, 0.7])

        for loop_closure_t_id in tile_info.potential_loop_closure:
            edges.append([index_list.index(tile_info_index), index_list.index(loop_closure_t_id)])
            edge_colors.append([0.8, 0.8, 1])

    edge_set.points = Vector3dVector(tile_positions)
    edge_set.lines = Vector2iVector(edges)
    edge_set.colors = Vector3dVector(edge_colors)

    # make sensor data point cloud


    draw_list = []
    if show_point:
        draw_list.append(tile_info_point_cloud)
    if show_edge:
        draw_list.append(edge_set)
    if show_key_frame:
        draw_list += tile_key_frame_list
    if show_regular_frame:
        draw_list += tile_regular_frame_list
    if show_senor_point:
        draw_list.append(sensor_point_cloud)
        draw_list.append(sensor_edge_set)

    draw_geometries(draw_list)
