from open3d import *
import numpy
from os.path import *
import cv2

import logging
import threading
import time


def make_tile_frame(trans_matrix, width=0.0052, height=0.0039, color=[0.5, 0.5, 0.5], fill=False):
    tile_frame = LineSet()
    lb_rb_rt_lt = [[0, -width / 2, -height / 2],
                   [0, width / 2, -height / 2],
                   [0, width / 2,  height / 2],
                   [0, -width / 2,  height / 2]
                   ]
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

        self.tile_width = 0.0048
        self.tile_height = 0.0036

        self.tile_counter = 0
        self.tile_pose_dict = {}
        self.tile_wire_frame_dict = {}

        self.viewer.create_window()
        # self.viewer.add_geometry(self.coordinate_frame)
        # self.viewer.add_geometry(self.camera_wire_frames_controller.wire_frame)
        # self.viewer.run()

        self.img_directory_path = ""
        self.info_directory_path = ""

        # self.viewer_window_thread = threading.Thread(target=self.keep_updating)

    # def update_camera_pose_robotic(self, pose_rob=numpy.ones(16)):
    #     self.current_camera_pose = rob_pose_to_trans(pose_rob)
    #     self.camera_wire_frames_controller.update_pose(self.current_camera_pose)

    def add_sampled_tile_wire_frame(self, pose=None):
        if pose is None:
            pose = self.current_camera_pose
        new_tile_wire_frame = make_tile_frame(pose,
                                              self.tile_width, self.tile_height, color=[0.5, 0.5, 0.5])
        self.tile_wire_frame_dict[self.tile_counter] = new_tile_wire_frame
        self.viewer.add_geometry(self.tile_wire_frame_dict[self.tile_counter])

        # self.update_visualizer()

        self.tile_pose_dict[self.tile_counter] = pose

    def read_robotic_arm_pose_list(self, robotic_pose_list_path):
        # robotic_pose_list = json.load(open(robotic_pose_list_path, "r"))["pose_list"]
        robotic_pose_list = json.load(open(robotic_pose_list_path, "r"))
        for i, robotic_pose in enumerate(robotic_pose_list):
            print("Loading pose %d" % i)
            pose = rob_pose_to_trans(robotic_pose)
            self.add_sampled_tile_wire_frame(pose)

    # def save_tile_img(self, id=None, img_dir=None):
    #     ret, img = self.microscope_camera.read()
    #     if id is not None:
    #         file_name = "tile_%6d.png" % id
    #     else:
    #         file_name = "tile_%6d.png" % self.tile_counter
    #     if img_dir is None:
    #         img_dir = self.img_directory_path
    #     image_path = join(img_dir, file_name)
    #     cv2.imwrite(image_path, img)

    # def save_tile_json(self, id=None, json_dir=None):
    #     json_path = join()
    #
    #
    # def save_tile_dict_json(self, dict_json_path=None):

    def update_visualizer(self):
        self.viewer.update_geometry()
        self.viewer.poll_events()
        self.viewer.update_renderer()

    # def keep_updating(self):
    #     while True:
    #         self.update_visualizer()
    #         # print("Updating")

    def run(self):
        # self.viewer_window_thread = threading.Thread(target=self.keep_updating)
        # self.viewer_window_thread.run()
        self.viewer.run()

    # def remove_coord(self):
    #     self.viewer.remove_geometry(self.coordinate_frame)
    #     self.update_visualizer()
