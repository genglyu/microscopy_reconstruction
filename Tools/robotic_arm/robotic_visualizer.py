from open3d import *
import numpy
from os.path import *
import cv2

import logging
import threading
import time
from robotic_data_convert import *


def make_tile_frame(trans_matrix, width=0.0052, height=0.0039, color=[0.5, 0.5, 0.5], fill=False):
    tile_frame = LineSet()
    lb_rb_rt_lt = [[0, -width / 2, -height / 2],
                   [0, width / 2, -height / 2],
                   [0, width / 2,  height / 2],
                   [0, -width / 2,  height / 2]]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    colors = [color, color, color, color]
    tile_frame.points = Vector3dVector(lb_rb_rt_lt)
    tile_frame.lines = Vector2iVector(lines)
    tile_frame.colors = Vector3dVector(colors)
    tile_frame.transform(trans_matrix)
    return tile_frame


def make_pcd_from_trans_list(trans_list, color=[0,0,0]):
    pcd = PointCloud()
    points = []
    normals = []
    colors = []
    for trans in trans_list:
        points.append(numpy.dot(trans, numpy.asarray([0, 0, 0, 1]).T).T[0:3])
        normals.append(numpy.dot(trans, numpy.asarray([1, 0, 0, 0]).T).T[0:3])
        colors.append(color)
    pcd.points = Vector3dVector(points)
    pcd.normals = Vector3dVector(normals)
    pcd.colors = Vector3dVector(colors)
    return pcd


def make_tile_frame_list_from_trans_list(trans_list, width=0.0052, height=0.0039, color=[0, 0, 0]):
    tile_frame_list = []
    for trans in trans_list:
        tile_frame = make_tile_frame(trans, width, height, color)
        tile_frame_list.append(tile_frame)
    return tile_frame_list


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


class RoboticVisualizerOpen3d:
    def __init__(self, robotic_config):
        self.robotic_config = robotic_config
        # self.microscope_camera = cv2.VideoCapture(0)
        # self.viewer = Visualizer()
        self.coordinate_frame = geometry.create_mesh_coordinate_frame(size=0.01, origin=[0.0, 0.0, 0.0])

        self.tile_width = robotic_config["tile_width"]
        self.tile_height = robotic_config["tile_height"]

        self.original_pcd = None
        self.original_wire_frame = None
        self.aligned_pcd = None
        self.aligned_wire_frame = None
        self.interpolated_pcd = None
        self.interpolated_wire_frame = None

        self.navigated_route = None


    def view_via_robotic_config(self):
        for view_setting in self.robotic_config["visualization"]:
            self.view(original_pcd=view_setting["original_pcd"],
                      original_wire_frame=view_setting["original_wire_frame"],
                      aligned_pcd=view_setting["aligned_pcd"],
                      aligned_wire_frame=view_setting["aligned_wire_frame"],
                      interpolated_pcd=view_setting["interpolated_pcd"],
                      interpolated_wire_frame=view_setting["interpolated_wire_frame"],
                      navigated_route=view_setting["navigated_route"])


    def view(self,
             original_pcd=False,
             original_wire_frame=False,
             aligned_pcd=False,
             aligned_wire_frame=False,
             interpolated_pcd=False,
             interpolated_wire_frame=False,
             navigated_route=False):
        draw_list =[]
        if original_pcd:
            if self.original_pcd is None:
                original_trans_list = read_trans_list(join(self.robotic_config["path_data"],
                                                           self.robotic_config["robotic_reconstruction_workspace"],
                                                           self.robotic_config["robotic_reconstruction_trans_list_all"])
                                                      )
                self.original_pcd = make_pcd_from_trans_list(original_trans_list, color=[0, 0, 0])
            draw_list.append(self.original_pcd)
        if original_wire_frame:
            if self.original_wire_frame is None:
                original_trans_list = read_trans_list(join(self.robotic_config["path_data"],
                                                           self.robotic_config["robotic_reconstruction_workspace"],
                                                           self.robotic_config["robotic_reconstruction_trans_list_all"])
                                                      )
                self.original_wire_frame = make_tile_frame_list_from_trans_list(original_trans_list)
            draw_list += self.original_wire_frame
        if aligned_pcd:
            if self.aligned_pcd is None:
                aligned_trans_list = read_trans_list(
                    join(self.robotic_config["path_data"],
                         self.robotic_config["robotic_reconstruction_workspace"],
                         self.robotic_config["robotic_reconstruction_trans_list_aligned"])
                )
                self.aligned_pcd = make_pcd_from_trans_list(aligned_trans_list,
                                                            color=[1, 0, 0])
            draw_list.append(self.aligned_pcd)
        if aligned_wire_frame:
            if self.aligned_wire_frame is None:
                aligned_trans_list = read_trans_list(
                    join(self.robotic_config["path_data"],
                         self.robotic_config["robotic_reconstruction_workspace"],
                         self.robotic_config["robotic_reconstruction_trans_list_aligned"])
                )
                self.aligned_wire_frame = make_tile_frame_list_from_trans_list(aligned_trans_list,
                                                                               color=[0.7, 0.7, 0.7])
            draw_list += self.aligned_wire_frame
        if interpolated_pcd:
            if self.interpolated_pcd is None:
                interpolated_trans_list = read_trans_list(
                    join(self.robotic_config["path_data"],
                         self.robotic_config["robotic_reconstruction_workspace"],
                         self.robotic_config["robotic_reconstruction_trans_interpolated"])
                )
                self.interpolated_pcd = make_pcd_from_trans_list(interpolated_trans_list, color=[0, 1, 0])
            draw_list.append(self.interpolated_pcd)
        if interpolated_wire_frame:
            if self.interpolated_wire_frame is None:
                interpolated_trans_list = read_trans_list(
                    join(self.robotic_config["path_data"],
                         self.robotic_config["robotic_reconstruction_workspace"],
                         self.robotic_config["robotic_reconstruction_trans_interpolated"])
                )
                self.interpolated_wire_frame = make_tile_frame_list_from_trans_list(interpolated_trans_list)
            draw_list += self.interpolated_wire_frame
        if navigated_route:
            if self.navigated_route is None:
                ordered_trans_list = read_trans_list(
                    join(self.robotic_config["path_data"],
                         self.robotic_config["robotic_reconstruction_workspace"],
                         self.robotic_config["robotic_reconstruction_trans_ordered"])
                )
                navigated_pcd = make_pcd_from_trans_list(ordered_trans_list)
                self.navigated_route = make_connection_of_pcd_order(navigated_pcd, color=[1, 0, 0])
            draw_list.append(self.navigated_route)

        draw_geometries(draw_list)
