from open3d import *
import numpy
import sys
sys.path.append("./Utility")
sys.path.append("./CoreModules")
sys.path.append("./Tools")
from tile_info_processing import *
from image_processing import *


# ======================================================================================================================
# openCV involved in these two functions.
def make_full_image_pcd_list_pose(tile_info_dict, downsample_factor=-1.0, color_filter=[1.0, 1.0, 1.0]):
    pcd_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        img = cv2.imread(tile_info.image_path)
        pcd = load_image_as_planar_point_cloud_open3d(image_bgr=img,
                                                      width_by_mm=tile_info.width_by_mm,
                                                      height_by_mm=tile_info.height_by_mm,
                                                      cv_scale_factor=downsample_factor,
                                                      color_filter=color_filter)
        pcd.transform(tile_info.pose_matrix)
        pcd_list.append(pcd)
    return pcd_list


def make_full_image_pcd_list_sensor(tile_info_dict, downsample_factor=-1.0, color_filter=[1.0, 1.0, 1.0]):
    pcd_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        img = cv2.imread(tile_info.image_path)
        pcd = load_image_as_planar_point_cloud_open3d(image_bgr=img,
                                                      width_by_mm=tile_info.width_by_mm,
                                                      height_by_mm=tile_info.height_by_mm,
                                                      cv_scale_factor=downsample_factor,
                                                      color_filter=color_filter)
        pcd.transform(tile_info.init_transform_matrix)
        pcd_list.append(pcd)
    return pcd_list
# ======================================================================================================================

# ======================================================================================================================
# For functions below, only open3d is used for visualization
# ======================================================================================================================


def make_connection_of_pcd_order(pcd, color=[0, 0, 0]):
    connection = LineSet()
    connection.points = pcd.points
    lines = []
    colors = []
    for i, point in enumerate(pcd.points):
        if i > 0:
            lines.append([i-1, i])
            colors.append(color)
    connection.lines = Vector2iVector(lines)
    connection.colors = Vector3dVector(colors)
    return connection


def make_point_cloud(points, color=[0.0, 0.0, 0.0], normals=None):
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    pcd.colors = Vector3dVector(numpy.repeat([color], len(points), axis=0))
    if normals is not None:
        pcd.normals = Vector3dVector(normals)
        pcd.normalize_normals()
    return pcd


def make_edge_set(points, edges, color=[0.0, 0.0, 0.0]):
    edge_set = LineSet()
    edge_set.points = Vector3dVector(points)
    edge_set.lines = Vector2iVector(edges)
    edge_set.colors = Vector3dVector(numpy.repeat([color], len(edges), axis=0))
    return edge_set


def make_pose_sensor_edge_set(tile_info_dict, color=[0.0, 0.0, 0.0]):
    """:returns points as List[n, 3] cause it's only used for sensor edges"""
    points = []
    edges = []
    for i, tile_info_key in enumerate(tile_info_dict):
        points.append(numpy.dot(tile_info_dict[tile_info_key].pose_matrix,
                                numpy.asarray([0, 0, 0, 1]).T).T[0:3])
        points.append(numpy.dot(tile_info_dict[tile_info_key].init_transform_matrix,
                                numpy.asarray([0, 0, 0, 1]).T).T[0:3])
        edges.append([i*2, i*2+1])
    edge_set = make_edge_set(points, edges, color)
    return edge_set


def make_key_frame_wireframes_pose(tile_info_dict, color=[0.0, 0.0, 0.0]):
    """:returns list of transformed rectangles made of open3d.LineSet"""
    tile_wireframe_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        if tile_info.is_keyframe:
            tile_wireframe_list.append(make_tile_frame(trans_matrix=tile_info.pose_matrix,
                                                       width_by_mm=tile_info.width_by_mm,
                                                       height_by_mm=tile_info.height_by_mm,
                                                       color=color))
    return tile_wireframe_list


def make_regular_frame_wireframes_pose(tile_info_dict, color=[0.0, 0.0, 0.0]):
    """:returns list of transformed rectangles made of open3d.LineSet"""
    tile_wireframe_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        if not tile_info.is_keyframe:
            tile_wireframe_list.append(make_tile_frame(trans_matrix=tile_info.pose_matrix,
                                                       width_by_mm=tile_info.width_by_mm,
                                                       height_by_mm=tile_info.height_by_mm,
                                                       color=color))
    return tile_wireframe_list


def make_key_frame_wireframes_sensor(tile_info_dict, color=[0.0, 0.0, 0.0]):
    """:returns list of transformed rectangles made of open3d.LineSet"""
    tile_wireframe_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        if tile_info.is_keyframe:
            tile_wireframe_list.append(make_tile_frame(trans_matrix=tile_info.init_transform_matrix,
                                                       width_by_mm=tile_info.width_by_mm,
                                                       height_by_mm=tile_info.height_by_mm,
                                                       color=color))
    return tile_wireframe_list


def make_regular_frame_wireframes_sensor(tile_info_dict, color=[0.0, 0.0, 0.0]):
    """:returns list of transformed rectangles made of open3d.LineSet"""
    tile_wireframe_list = []
    for tile_info_key in tile_info_dict:
        tile_info = tile_info_dict[tile_info_key]
        if not tile_info.is_keyframe:
            tile_wireframe_list.append(make_tile_frame(trans_matrix=tile_info.init_transform_matrix,
                                                       width_by_mm=tile_info.width_by_mm,
                                                       height_by_mm=tile_info.height_by_mm,
                                                       color=color))
    return tile_wireframe_list
# =======================================================================================


def make_tile_frame(trans_matrix=numpy.identity(4),
                    width_by_mm=4.0, height_by_mm=3.0, color=[0.5, 0.5, 0.5]):
    tile_frame = LineSet()
    lb_rb_rt_lt = [[0, -width_by_mm / 2, -height_by_mm / 2],
                   [0,  width_by_mm / 2, -height_by_mm / 2],
                   [0,  width_by_mm / 2,  height_by_mm / 2],
                   [0, -width_by_mm / 2,  height_by_mm / 2]
                   ]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    colors = [color, color, color, color]

    tile_frame.points = Vector3dVector(lb_rb_rt_lt)
    tile_frame.lines = Vector2iVector(lines)
    tile_frame.colors = Vector3dVector(colors)
    tile_frame.transform(trans_matrix)

    return tile_frame


def generate_pose_points_and_normals(tile_info_dict, is_key_frame=None):
    """:returns points_and_normals = {"points": [],"normals": []}"""
    points_and_normals = {"points": [],
                          "normals": []}
    if is_key_frame is None:
        for tile_info_key in tile_info_dict:
            points_and_normals["points"].append(numpy.dot(tile_info_dict[tile_info_key].pose_matrix,
                                                          numpy.asarray([0, 0, 0, 1]).T).T[0:3])
            points_and_normals["normals"].append(numpy.dot(tile_info_dict[tile_info_key].pose_matrix,
                                                           numpy.asarray([1, 0, 0, 0]).T).T[0:3])
    else:
        for tile_info_key in tile_info_dict:
            if tile_info_dict[tile_info_key].is_keyframe == is_key_frame:
                points_and_normals["points"].append(numpy.dot(tile_info_dict[tile_info_key].pose_matrix,
                                                              numpy.asarray([0, 0, 0, 1]).T).T[0:3])
                points_and_normals["normals"].append(numpy.dot(tile_info_dict[tile_info_key].pose_matrix,
                                                              numpy.asarray([1, 0, 0, 0]).T).T[0:3])
    return points_and_normals


def generate_sensor_points_and_normals(tile_info_dict, is_key_frame=None):
    """:returns points_and_normals = {"points": [],"normals": []}"""
    points_and_normals = {"points": [],
                          "normals": []}
    if is_key_frame is None:
        for tile_info_key in tile_info_dict:
            points_and_normals["points"].append(numpy.dot(tile_info_dict[tile_info_key].init_transform_matrix,
                                                          numpy.asarray([0, 0, 0, 1]).T).T[0:3])
            points_and_normals["normals"].append(numpy.dot(tile_info_dict[tile_info_key].init_transform_matrix,
                                                           numpy.asarray([1, 0, 0, 0]).T).T[0:3])
    else:
        for tile_info_key in tile_info_dict:
            if tile_info_dict[tile_info_key].is_keyframe == is_key_frame:
                points_and_normals["points"].append(numpy.dot(tile_info_dict[tile_info_key].init_transform_matrix,
                                                              numpy.asarray([0, 0, 0, 1]).T).T[0:3])
                points_and_normals["normals"].append(numpy.dot(tile_info_dict[tile_info_key].init_transform_matrix,
                                                               numpy.asarray([1, 0, 0, 0]).T).T[0:3])
    return points_and_normals


def generate_odometry_edges(tile_info_dict):
    key_list = list(tile_info_dict.keys())
    edges = []
    for tile_info_key in tile_info_dict:
        for odometry_key in tile_info_dict[tile_info_key].odometry_list:
            edges.append([key_list.index(tile_info_key), key_list.index(odometry_key)])
    return edges


def generate_confirmed_loop_closure_edges(tile_info_dict):
    key_list = list(tile_info_dict.keys())
    edges = []
    for tile_info_key in tile_info_dict:
        for confirmed_loop_closure_key in tile_info_dict[tile_info_key].confirmed_loop_closure:
            edges.append([key_list.index(tile_info_key), key_list.index(confirmed_loop_closure_key)])
    return edges


def generate_false_loop_closure_edges(tile_info_dict):
    key_list = list(tile_info_dict.keys())
    edges = []
    for tile_info_key in tile_info_dict:
        for potential_loop_closure_key in tile_info_dict[tile_info_key].potential_loop_closure:
            if potential_loop_closure_key not in tile_info_dict[tile_info_key].confirmed_loop_closure:
                edges.append([key_list.index(tile_info_key), key_list.index(potential_loop_closure_key)])
    return edges

# ========================================================================================


class MicroscopyReconstructionVisualizerOpen3d:
    def __init__(self, tile_info_dict, config):

        self.tile_info_dict = tile_info_dict
        self.config = config

        self.pose_kf_pan = generate_pose_points_and_normals(tile_info_dict, is_key_frame=True)
        self.pose_rf_pan = generate_pose_points_and_normals(tile_info_dict, is_key_frame=False)
        self.sensor_kf_pan = generate_sensor_points_and_normals(tile_info_dict, is_key_frame=True)
        self.sensor_rf_pan = generate_sensor_points_and_normals(tile_info_dict, is_key_frame=False)

        self.pose_pan = generate_pose_points_and_normals(tile_info_dict)
        self.sensor_pan = generate_sensor_points_and_normals(tile_info_dict)

        self.odo_edges = generate_odometry_edges(tile_info_dict)
        self.c_loop_edges = generate_confirmed_loop_closure_edges(tile_info_dict)
        self.f_loop_edges = generate_false_loop_closure_edges(tile_info_dict)
        # ==================================================
        self.key_frame_pcd_pose = None
        self.regular_frame_pcd_pose = None

        self.key_frame_wireframes_pose = None
        self.regular_frame_wireframes_pose = None

        self.odometry_edgeset_pose = None
        self.c_loop_edgeset_pose = None
        self.f_loop_edgeset_pose = None

        self.full_image_pcd_list_pose = None
        # ==================================================
        self.key_frame_pcd_sensor = None
        self.regular_frame_pcd_sensor = None

        self.key_frame_wireframes_sensor = None
        self.regular_frame_wireframes_sensor = None

        self.odometry_edgeset_sensor = None
        self.c_loop_edgeset_sensor = None
        self.f_loop_edgeset_sensor = None

        self.full_image_pcd_list_sensor = None

        self.sensor_edgeset = None

    def visualize_config(self):
        for view_setting in self.config["visualization"]:
            self.view(
                key_frame_pcd_pose=view_setting["key_frame_pcd_pose_is_enabled"],
                regular_frame_pcd_pose=view_setting["regular_frame_pcd_pose_is_enabled"],
                key_frame_wireframes_pose=view_setting["key_frame_wireframes_pose_is_enabled"],
                regular_frame_wireframes_pose=view_setting["regular_frame_wireframes_pose_is_enabled"],
                odometry_edgeset_pose=view_setting["odometry_edgeset_pose_is_enabled"],
                c_loop_edgeset_pose=view_setting["c_loop_edgeset_pose_is_enabled"],
                f_loop_edgeset_pose=view_setting["f_loop_edgeset_pose_is_enabled"],
                full_image_pcd_pose=view_setting["full_image_pcd_pose_is_enabled"],
                key_frame_pcd_sensor=view_setting["key_frame_pcd_sensor_is_enabled"],
                regular_frame_pcd_sensor=view_setting["regular_frame_pcd_sensor_is_enabled"],
                key_frame_wireframes_sensor=view_setting["key_frame_wireframes_sensor_is_enabled"],
                regular_frame_wireframes_sensor=view_setting["regular_frame_wireframes_sensor_is_enabled"],
                odometry_edgeset_sensor=view_setting["odometry_edgeset_sensor_is_enabled"],
                c_loop_edgeset_sensor=view_setting["c_loop_edgeset_sensor_is_enabled"],
                f_loop_edgeset_sensor=view_setting["f_loop_edgeset_sensor_is_enabled"],
                full_image_pcd_sensor=view_setting["full_image_pcd_sensor_is_enabled"],
                sensor_edgeset=view_setting["sensor_edgeset_is_enabled"],
                image_pcd_downsample_factor=view_setting["image_pcd_downsample_factor"]
            )


    def view(self,
             key_frame_pcd_pose=False,
             regular_frame_pcd_pose=False,

             key_frame_wireframes_pose=False,
             regular_frame_wireframes_pose=False,

             odometry_edgeset_pose=False,
             c_loop_edgeset_pose=False,
             f_loop_edgeset_pose=False,

             full_image_pcd_pose=False,

             key_frame_pcd_sensor=False,
             regular_frame_pcd_sensor=False,

             key_frame_wireframes_sensor=False,
             regular_frame_wireframes_sensor=False,

             odometry_edgeset_sensor=False,
             c_loop_edgeset_sensor=False,
             f_loop_edgeset_sensor=False,
             full_image_pcd_sensor=False,

             sensor_edgeset=False,
             image_pcd_downsample_factor=0.1):

        draw_list = []

        if key_frame_pcd_pose:
            if self.key_frame_pcd_pose is None:
                self.key_frame_pcd_pose = make_point_cloud(points=self.pose_kf_pan["points"],
                                                           color=self.config["key_frame_pcd_pose_color"],
                                                           normals=self.pose_kf_pan["normals"])
            draw_list += [self.key_frame_pcd_pose]

        if regular_frame_pcd_pose:
            if self.regular_frame_pcd_pose is None:
                self.regular_frame_pcd_pose = make_point_cloud(points=self.pose_rf_pan["points"],
                                                               color=self.config["regular_frame_pcd_pose_color"],
                                                               normals=self.pose_rf_pan["normals"])
            draw_list += [self.regular_frame_pcd_pose]

        if key_frame_wireframes_pose:
            if self.key_frame_wireframes_pose is None:
                self.key_frame_wireframes_pose = \
                    make_key_frame_wireframes_pose(tile_info_dict=self.tile_info_dict,
                                                   color=self.config["key_frame_wireframes_pose_color"])
            draw_list += self.key_frame_wireframes_pose

        if regular_frame_wireframes_pose:
            if self.regular_frame_wireframes_pose is None:
                self.regular_frame_wireframes_pose = \
                    make_regular_frame_wireframes_pose(tile_info_dict=self.tile_info_dict,
                                                       color=self.config["regular_frame_wireframes_pose_color"])
            draw_list += self.regular_frame_wireframes_pose

        if odometry_edgeset_pose:
            if self.odometry_edgeset_pose is None:
                self.odometry_edgeset_pose = make_edge_set(points=self.pose_pan["points"],
                                                           edges=self.odo_edges,
                                                           color=self.config["odometry_edgeset_pose_color"])
            draw_list += [self.odometry_edgeset_pose]

        if c_loop_edgeset_pose:
            if self.c_loop_edgeset_pose is None:
                self.c_loop_edgeset_pose = make_edge_set(points=self.pose_pan["points"],
                                                         edges=self.c_loop_edges,
                                                         color=self.config["c_loop_edgeset_pose_color"])
            draw_list += [self.c_loop_edgeset_pose]

        if f_loop_edgeset_pose:
            if self.f_loop_edgeset_pose is None:
                self.f_loop_edgeset_pose = make_edge_set(points=self.pose_pan["points"],
                                                         edges=self.f_loop_edges,
                                                         color=self.config["f_loop_edgeset_pose_color"])
            draw_list += [self.f_loop_edgeset_pose]

        if full_image_pcd_pose:
            if self.full_image_pcd_list_pose is None:
                self.full_image_pcd_list_pose = \
                    make_full_image_pcd_list_pose(tile_info_dict=self.tile_info_dict,
                                                  downsample_factor=image_pcd_downsample_factor,
                                                  color_filter=self.config["full_image_pcd_pose_color_filter"]
                                                  )
            draw_list += self.full_image_pcd_list_pose

        if key_frame_pcd_sensor:
            if self.key_frame_pcd_sensor is None:
                self.key_frame_pcd_sensor = make_point_cloud(points=self.sensor_kf_pan["points"],
                                                           color=self.config["key_frame_pcd_sensor_color"],
                                                           normals=self.sensor_kf_pan["normals"])
            draw_list += [self.key_frame_pcd_sensor]

        if regular_frame_pcd_sensor:
            if self.regular_frame_pcd_sensor is None:
                self.regular_frame_pcd_sensor = make_point_cloud(points=self.sensor_rf_pan["points"],
                                                               color=self.config["regular_frame_pcd_sensor_color"],
                                                               normals=self.sensor_rf_pan["normals"])
            draw_list += [self.regular_frame_pcd_sensor]

        if key_frame_wireframes_sensor:
            if self.key_frame_wireframes_sensor is None:
                self.key_frame_wireframes_sensor = \
                    make_key_frame_wireframes_sensor(tile_info_dict=self.tile_info_dict,
                                                   color=self.config["key_frame_wireframes_sensor_color"])
            draw_list += self.key_frame_wireframes_sensor

        if regular_frame_wireframes_sensor:
            if self.regular_frame_wireframes_sensor is None:
                self.regular_frame_wireframes_sensor = \
                    make_regular_frame_wireframes_sensor(tile_info_dict=self.tile_info_dict,
                                                       color=self.config["regular_frame_wireframes_sensor_color"])
            draw_list += self.regular_frame_wireframes_sensor

        if odometry_edgeset_sensor:
            if self.odometry_edgeset_sensor is None:
                self.odometry_edgeset_sensor = make_edge_set(points=self.sensor_pan["points"],
                                                           edges=self.odo_edges,
                                                           color=self.config["odometry_edgeset_sensor_color"])
            draw_list += [self.odometry_edgeset_sensor]

        if c_loop_edgeset_sensor:
            if self.c_loop_edgeset_sensor is None:
                self.c_loop_edgeset_sensor = make_edge_set(points=self.sensor_pan["points"],
                                                         edges=self.c_loop_edges,
                                                         color=self.config["c_loop_edgeset_sensor_color"])
            draw_list += [self.c_loop_edgeset_sensor]

        if f_loop_edgeset_sensor:
            if self.f_loop_edgeset_sensor is None:
                self.f_loop_edgeset_sensor = make_edge_set(points=self.sensor_pan["points"],
                                                         edges=self.f_loop_edges,
                                                         color=self.config["f_loop_edgeset_sensor_color"])
            draw_list += [self.f_loop_edgeset_sensor]

        if full_image_pcd_sensor:
            if self.full_image_pcd_list_sensor is None:
                self.full_image_pcd_list_sensor = \
                    make_full_image_pcd_list_sensor(tile_info_dict=self.tile_info_dict,
                                                  downsample_factor=image_pcd_downsample_factor,
                                                  color_filter=self.config["full_image_pcd_sensor_color_filter"]
                                                  )
            draw_list += self.full_image_pcd_list_sensor

        if sensor_edgeset:
            if self.sensor_edgeset is None:
                self.sensor_edgeset = make_pose_sensor_edge_set(tile_info_dict=self.tile_info_dict,
                                                                color=self.config["sensor_edgeset_color"])
            draw_list += [self.sensor_edgeset]

        draw_geometries(draw_list)


def visualize_tile_info_dict_as_point_cloud(tile_info_dict, downsample_factor=-1):
    draw_list = []
    draw_list += make_full_image_pcd_list_pose(tile_info_dict, downsample_factor)
    draw_geometries(draw_list)


# def visualize_tile_info_dict(tile_info_dict,
#                              show_point=True, show_edge=True,
#                              show_key_frame=True, show_regular_frame=True,
#                              show_sensor_frame=False, show_senor_point=False):
#     index_list = []
#     # geometries would be used for drawing.
#     tile_key_frame_list = []
#     tile_regular_frame_list = []
#     tile_info_point_cloud = PointCloud()
#     edge_set = LineSet()
#     sensor_point_cloud = PointCloud()
#     sensor_edge_set = LineSet()
#
#     # Data used to make the geometries
#     tile_positions = []
#     tile_colors = []
#     tile_normals = []
#
#     sensor_positions = []
#     sensor_colors = []
#     tile_sensor_pos = []
#     sensor_edges = []
#     sensor_edge_colors = []
#
#     edges = []
#     edge_colors = []
#
#     # make the point cloud of nodes.
#     for tile_info_index in tile_info_dict:
#         tile_info = tile_info_dict[tile_info_index]
#         index_list.append(tile_info_index)
#
#         tile_position = numpy.dot(tile_info.pose_matrix, numpy.asarray([0, 0, 0, 1]).T).T[0:3]
#         sensor_position = numpy.dot(tile_info.init_transform_matrix, numpy.asarray([0, 0, 0, 1]).T).T[0:3]
#
#         tile_positions.append(tile_position)
#         sensor_positions.append(sensor_position)
#
#         tile_sensor_pos.append(tile_position)
#         tile_sensor_pos.append(sensor_position)
#
#         sensor_edges.append([index_list.index(tile_info_index) * 2, index_list.index(tile_info_index) * 2 + 1])
#         sensor_edge_colors.append([0.8, 0.8, 0.8])
#
#         if tile_info.is_keyframe:
#             tile_colors.append([1, 0, 0])
#             sensor_colors.append([0.7, 0.7, 1.0])
#             tile_key_frame_list.append(make_tile_frame(trans_matrix=tile_info.pose_matrix,
#                                                        width_by_mm=tile_info.width_by_mm,
#                                                        height_by_mm=tile_info.height_by_mm,
#                                                        color=[1, 0.8, 0.8]))
#         else:
#             tile_colors.append([0, 0, 0])
#             sensor_colors.append([0.7, 0.7, 1.0])
#             tile_regular_frame_list.append(make_tile_frame(trans_matrix=tile_info.pose_matrix,
#                                                            width_by_mm=tile_info.width_by_mm,
#                                                            height_by_mm=tile_info.height_by_mm,
#                                                            color=[0.8, 0.8, 0.8]))
#         tile_normals.append(
#             numpy.dot(tile_info.pose_matrix, numpy.asarray([0, 0, 1, 0]).T).T[0:3]
#         )
#     tile_info_point_cloud.points = Vector3dVector(tile_positions)
#     tile_info_point_cloud.colors = Vector3dVector(tile_colors)
#     tile_info_point_cloud.normals = Vector3dVector(tile_normals)
#     tile_info_point_cloud.normalize_normals()
#
#     sensor_point_cloud.points = Vector3dVector(sensor_positions)
#     sensor_point_cloud.colors = Vector3dVector(sensor_colors)
#
#     sensor_edge_set.points = Vector3dVector(tile_sensor_pos)
#     sensor_edge_set.lines = Vector2iVector(sensor_edges)
#     sensor_edge_set.colors = Vector3dVector(sensor_edge_colors)
#
#     # make outline frame
#     for tile_info_index in tile_info_dict:
#         tile_info = tile_info_dict[tile_info_index]
#         for odometry_t_id in tile_info.odometry_list:
#             edges.append([index_list.index(tile_info_index), index_list.index(odometry_t_id)])
#             if index_list.index(tile_info_index) + 1 == index_list.index(odometry_t_id):
#                 edge_colors.append([1, 0.7, 0.7])
#             else:
#                 edge_colors.append([0.7, 1, 0.7])
#
#         for loop_closure_t_id in tile_info.potential_loop_closure:
#             edges.append([index_list.index(tile_info_index), index_list.index(loop_closure_t_id)])
#             edge_colors.append([0.8, 0.8, 1])
#
#     edge_set.points = Vector3dVector(tile_positions)
#     edge_set.lines = Vector2iVector(edges)
#     edge_set.colors = Vector3dVector(edge_colors)
#
#     # make sensor data point cloud
#
#     draw_list = []
#     if show_point:
#         draw_list.append(tile_info_point_cloud)
#     if show_edge:
#         draw_list.append(edge_set)
#     if show_key_frame:
#         draw_list += tile_key_frame_list
#     if show_regular_frame:
#         draw_list += tile_regular_frame_list
#     if show_senor_point:
#         draw_list.append(sensor_point_cloud)
#         draw_list.append(sensor_edge_set)
#
#     draw_geometries(draw_list)
