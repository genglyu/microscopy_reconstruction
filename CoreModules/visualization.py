from open3d import *
import numpy
import sys
sys.path.append("./Utility")
sys.path.append("./CoreModules")
sys.path.append("./Tools")
from tile_info_processing import *


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


def visualize_tile_info_dict(tile_info_dict, show_point=True, show_edge=True,
                             show_key_frame=True, show_regular_frame=True):
    index_list = []

    tile_key_frame_list = []
    tile_regular_frame_list = []
    tile_info_point_cloud = PointCloud()
    edge_set = LineSet()

    tile_positions = []
    tile_colors = []
    tile_normals = []
    edges = []
    edge_colors = []

    for tile_info_index in tile_info_dict:
        tile_info = tile_info_dict[tile_info_index]
        index_list.append(tile_info_index)
        tile_positions.append(tile_info.position)
        if tile_info.is_keyframe:
            tile_colors.append([1, 0, 0])
            tile_key_frame_list.append(make_tile_frame(trans_matrix=tile_info.init_transform_matrix,
                                                       width_by_mm=tile_info.width_by_mm,
                                                       height_by_mm=tile_info.height_by_mm,
                                                       color=[1, 0.8, 0.8]))
        else:
            tile_colors.append([0, 0, 0])
            tile_regular_frame_list.append(make_tile_frame(trans_matrix=tile_info.init_transform_matrix,
                                                           width_by_mm=tile_info.width_by_mm,
                                                           height_by_mm=tile_info.height_by_mm,
                                                           color=[0.8, 0.8, 0.8]))
        tile_normals.append(
            numpy.dot(tile_info.init_transform_matrix, numpy.asarray([0, 0, 1, 0]).T).T[0:3]
        )

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

    tile_info_point_cloud.points = Vector3dVector(tile_positions)
    tile_info_point_cloud.colors = Vector3dVector(tile_colors)
    tile_info_point_cloud.normals = Vector3dVector(tile_normals)
    tile_info_point_cloud.normalize_normals()

    edge_set.points = Vector3dVector(tile_positions)
    edge_set.lines = Vector2iVector(edges)
    edge_set.colors = Vector3dVector(edge_colors)

    draw_list = []
    if show_point:
        draw_list.append(tile_info_point_cloud)
    if show_edge:
        draw_list.append(edge_set)
    if show_key_frame:
        draw_list += tile_key_frame_list
    if show_regular_frame:
        draw_list += tile_regular_frame_list

    draw_geometries(draw_list)
