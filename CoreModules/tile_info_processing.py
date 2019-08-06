import numpy
import os
import json
import xml.etree.ElementTree
import cv2
from scipy.spatial.transform import *
import transforms3d
import sys
from open3d import *
import math
import matplotlib.pyplot as plt

sys.path.append("../Utility")
sys.path.append("../Tools/robotic_arm")
from file_managing import *
from robotic_data_convert import *


class TileInfo:
    def __init__(self,
                 tile_index,
                 tile_info_path,
                 image_path,
                 zoom_level,
                 position=numpy.array([0.0, 0.0, 0.0]),
                 rotation=numpy.array([0.0, 0.0, 1.0, 0.0]),
                 init_transform_matrix=numpy.identity(4),
                 width_by_mm=5.2,
                 height_by_mm=3.9):
        self.tile_index = tile_index
        self.tile_info_path = tile_info_path
        self.image_path = image_path

        self.zoom_level = zoom_level

        self.position = position
        self.rotation = rotation

        # will be caculate later

        self.laplacian = 0.0
        self.is_keyframe = False
        self.width_by_pixel = 640
        self.height_by_pixel = 480
        self.width_by_mm = width_by_mm
        self.height_by_mm = height_by_mm

        self.has_april_tag = False
        self.april_tags = []
        self.trans_from_april_tag = {}

        self.pose_matrix = numpy.identity(4)
        self.init_transform_matrix = init_transform_matrix

        self.odometry_list = []
        self.potential_loop_closure = []  # Use the real tile index rather than key in dict. (key should be the same)
        self.confirmed_loop_closure = []


def tile_generate_init_transform_matrix(position, rotation, scale=[1, 1, 1]):  # After updating and normalize tile info.
    translation = position
    rotation_matrix = Rotation.from_quat(rotation).as_dcm()
    init_transform_matrix = transforms3d.affines.compose(translation, rotation_matrix, scale)
    # Because of using x+ as normal direction in file image_processing.py, a rotation needs to be added.
    # plane_shifting_rotation_trans = \
    #     transforms3d.affines.compose(T=[0, 0, 0],
    #                                  R=Rotation.from_euler("xyz", [0, -math.pi/2, -math.pi/2]).as_dcm(),
    #                                  Z=[1, 1, 1])
    plane_shifting_rotation_trans = numpy.array([[0, 1, 0, 0],
                                                 [0, 0, 1, 0],
                                                 [1, 0, 0, 0],
                                                 [0, 0, 0, 1]])
    init_transform_matrix = numpy.dot(init_transform_matrix, plane_shifting_rotation_trans)
    # =============================================================================================
    return init_transform_matrix


def load_tile_info_unity(tile_info_file_path):
    # print("Try loading from " + tile_info_file_path)
    tree = xml.etree.ElementTree.parse(tile_info_file_path)
    tile_info_element = tree.getroot()
    # Try to get the real index from files.
    try:
        tile_index = int(tile_info_element.get('index'))
    except:
        file_name, extension = os.path.splitext(os.path.basename(tile_info_file_path))
        tile_index = int(file_name[len("tile_"):])

    new_tile_info = TileInfo(
        tile_index=tile_index,
        tile_info_path=tile_info_file_path,
        image_path=tile_info_element.find('imagePath').text,

        zoom_level=int(tile_info_element.get('zoomLevel')),
        position=numpy.array([
            float(tile_info_element.find('position').get('x')),
            float(tile_info_element.find('position').get('y')),
            -float(tile_info_element.find('position').get('z'))]) * 1000,
        # Orders of w x y z are different in different libs.
        rotation=numpy.array([
            float(tile_info_element.find('rotation').get('x')),
            float(tile_info_element.find('rotation').get('y')),
            -float(tile_info_element.find('rotation').get('z')),
            -float(tile_info_element.find('rotation').get('w'))
        ])
    )
    return new_tile_info


# def load_tile_info_robotic(tile_info_file_path):
#     if os.path.isfile(tile_info_file_path):
#         robotic_pose_data = json.load(open(tile_info_file_path, "r"))
#         trans_data = rob_pose_to_trans(robotic_pose_data)
#
#         file_name, extension = os.path.splitext(os.path.basename(tile_info_file_path))
#         tile_index = int(file_name[len("tile_"):])
#
#         new_tile_info = TileInfo(
#             tile_index=tile_index,
#
#         )


# def make_tile_info_dict_all_robotic(config):
#     tile_info_directory = join(config["path_data"], config["path_tile_info"])
#     if os.path.isdir(tile_info_directory):
#         tile_info_file_list = get_file_list(tile_info_directory)
#     else:
#         print(tile_info_directory + " is not a directory")
#         return None
#     # Usually there is no need to add the sorting function
#
#     # camera offset calibration would use these two. Might need to be adjusted in the future. ========================
#     camera_offset_matrix = numpy.asarray(config["camera_offset"]).reshape((4, 4))
#     camera_offset_matrix_inv = numpy.linalg.inv(camera_offset_matrix)
#     # ================================================================================================================
#     tile_info_dict_all = {}
#     # tile_laplacian_list = []
#     tile_blur = 0
#
#     for tile_info_file_path in tile_info_file_list:
#         tile_info = load_tile_info_unity(tile_info_file_path)
#         tile_info.image_path = os.path.join(config["path_data"], config["path_image_dir"], tile_info.image_path)
#
#         image = cv2.imread(tile_info.image_path)
#         (h, w, c) = image.shape
#         [tile_info.width_by_pixel, tile_info.height_by_pixel] = [w, h]
#         [tile_info.width_by_mm, tile_info.height_by_mm] = config["size_by_mm"][tile_info.zoom_level]
#
#         tile_info.laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
#         # tile_laplacian_list.append(tile_info.laplacian)
#
#         tile_info.init_transform_matrix = tile_generate_init_transform_matrix(tile_info.position, tile_info.rotation)
#
#         tile_info.pose_matrix = tile_info.init_transform_matrix
#         tile_info_dict_all[tile_info.tile_index] = tile_info
#         # ===========================================================================================
#         if tile_info.laplacian < config["laplacian_threshold"]:
#             tile_blur += 1
#             print("Blur: %05d. Total blurs: %d out of %d" % (tile_info.tile_index, tile_blur, len(tile_info_file_list)))
#
#     print("%d tiles out of %d in total are blurred" % (tile_blur, len(tile_info_dict_all)))
#
#     # plt.hist(x=tile_laplacian_list)
#     # # plt.hist(x=tile_laplacian_list, bins=numpy.arange(-5.0, 5.5, 0.5))
#     # plt.title('The laplacian of tile images')
#     # plt.xlabel('laplacian')
#     # plt.ylabel('Distribution')
#     # # plt.grid(axis='y', alpha=0.5)
#     # # axes = plt.gca()
#     # # axes.set_xlim([-5, 5])
#     # # axes.set_ylim([0, 3000000])
#     # plt.show()
#
#     return tile_info_dict_all


def make_tile_info_dict_all_unity(config):
    tile_info_directory = join(config["path_data"], config["path_tile_info"])
    if os.path.isdir(tile_info_directory):
        tile_info_file_list = get_file_list(tile_info_directory)
    else:
        print(tile_info_directory + " is not a directory")
        return None
    # Usually there is no need to add the sorting function

    # camera offset calibration would use these two. Might need to be adjusted in the future. ========================
    camera_offset_matrix = numpy.asarray(config["camera_offset"]).reshape((4, 4))
    camera_offset_matrix_inv = numpy.linalg.inv(camera_offset_matrix)
    # ================================================================================================================

    tile_info_dict_all = {}
    # tile_laplacian_list = []
    tile_blur = 0

    for tile_info_file_path in tile_info_file_list:
        tile_info = load_tile_info_unity(tile_info_file_path)
        tile_info.image_path = os.path.join(config["path_data"], config["path_image_dir"], tile_info.image_path)

        image = cv2.imread(tile_info.image_path)
        (h, w, c) = image.shape
        [tile_info.width_by_pixel, tile_info.height_by_pixel] = [w, h]
        [tile_info.width_by_mm, tile_info.height_by_mm] = config["size_by_mm"][tile_info.zoom_level]

        tile_info.laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
        # tile_laplacian_list.append(tile_info.laplacian)

        tile_info.init_transform_matrix = tile_generate_init_transform_matrix(tile_info.position, tile_info.rotation)

        # camera offset calibration. Might need to be adjusted in the future.========================
        # current matrix is generated for normal direction == z+. Might not suit for other situation.

        # tile_info.init_transform_matrix = numpy.dot(
        #     numpy.dot(camera_offset_matrix_inv, tile_info.init_transform_matrix),
        #     camera_offset_matrix)

        # ===========================================================================================

        tile_info.pose_matrix = tile_info.init_transform_matrix
        tile_info_dict_all[tile_info.tile_index] = tile_info
        # ===========================================================================================
        if tile_info.laplacian < config["laplacian_threshold"]:
            tile_blur += 1
            print("Blur: %05d. Total blurs: %d out of %d" % (tile_info.tile_index, tile_blur, len(tile_info_file_list)))

    print("%d tiles out of %d in total are blurred" % (tile_blur, len(tile_info_dict_all)))

    # plt.hist(x=tile_laplacian_list)
    # # plt.hist(x=tile_laplacian_list, bins=numpy.arange(-5.0, 5.5, 0.5))
    # plt.title('The laplacian of tile images')
    # plt.xlabel('laplacian')
    # plt.ylabel('Distribution')
    # # plt.grid(axis='y', alpha=0.5)
    # # axes = plt.gca()
    # # axes.set_xlim([-5, 5])
    # # axes.set_ylim([0, 3000000])
    # plt.show()

    return tile_info_dict_all


def make_info_dict(tile_info_dict_all, config):
    tile_info_dict_subsample = {}

    previous_laplacian = 0
    for i, tile_info_key in enumerate(tile_info_dict_all):
        tile_info = tile_info_dict_all[tile_info_key]
        if (config["index_start"] <= tile_info.tile_index <= config["index_end"]
            or config["index_start"] == -1 or config["index_end"] == -1) \
                and i % config["subsample_per_n_frame"] == 0:
            if tile_info.laplacian > config["laplacian_threshold"] \
                    and tile_info.laplacian > previous_laplacian * config["laplacian_relative_threshold"]:
                previous_laplacian = tile_info.laplacian
                tile_info_dict_subsample[tile_info_key] = tile_info

    # Recenter the pos and rotation with references ================================================
    # To simplify, use the first tile as reference.
    reference_trans_matrix_inv = numpy.identity(4)

    for j, tile_info_key in enumerate(tile_info_dict_subsample):
        tile_info = tile_info_dict_subsample[tile_info_key]
        if j == 0:
            reference_trans_matrix_inv = numpy.linalg.inv(tile_info.init_transform_matrix)
        # if j % config["n_keyframes_per_n_frame"] == 0:
        #     tile_info.is_keyframe = True
        # else:
        #     tile_info.is_keyframe = False
        # recenter all the tiles to the reference tile.
        tile_info.init_transform_matrix = numpy.dot(reference_trans_matrix_inv, tile_info.init_transform_matrix)
        tile_info.pose_matrix = tile_info.init_transform_matrix
        (pt, pr, pz, ps) = transforms3d.affines.decompose44(tile_info.init_transform_matrix)

        tile_info.position = pt
        tile_info.rotation = Rotation.from_dcm(pr).as_quat()

    # Set key frames. ===============================================================================
    # Actually, this might not be the best way of choosing key frames. Down sampling makes the key
    # frames far away from each other.

    tile_info_point_cloud, tile_tree, tile_index_list = \
        tile_info_dict_generate_kd_tree(tile_info_dict_subsample)
    # pcd_down, index_in_pcd =\
    #     geometry.voxel_down_sample_and_trace(input=tile_info_point_cloud,
    #                                          voxel_size=config["keyframe_voxel_size"],
    #                                          min_bound=numpy.asarray(config["keyframe_min_bound_T"]).T,
    #                                          max_bound=numpy.asarray(config["keyframe_max_bound_T"]).T,
    #                                          approximate_class=False)
    # index_list = index_in_pcd.reshape(-1)
    # index_list = numpy.sort(index_list[index_list >= 0]).tolist()
    # for index in index_list:
    #     tile_info_dict_subsample[tile_index_list[index]].is_keyframe = True

    for j, tile_info_key in enumerate(tile_info_dict_subsample):
        if j % config["n_keyframes_per_n_frame"] == 0:
            tile_info_dict_subsample[tile_info_key].is_keyframe = True

    # Generate potential neighbours. ================================================================
    for k, tile_info_key in enumerate(tile_info_dict_subsample):
        tile_info = tile_info_dict_subsample[tile_info_key]
        [_, idx, _] = \
            tile_tree.search_radius_vector_3d(tile_info.position,
                                              numpy.array(config["size_by_mm"])[0][0]
                                              * config["loop_closure_search_range_factor"])
        n_idx = numpy.array(idx)

        for index_in_list in n_idx:
            if tile_index_list[index_in_list] > tile_info.tile_index:
                potential_adjacent_tile_index = tile_index_list[index_in_list]

                if index_in_list - tile_index_list.index(tile_info_key) <= config["odometry_range"]:
                    tile_info.odometry_list.append(potential_adjacent_tile_index)
                else:
                    if tile_info.is_keyframe and tile_info_dict_subsample[potential_adjacent_tile_index].is_keyframe:
                        trans_target = tile_info.init_transform_matrix
                        trans_source = tile_info_dict_subsample[potential_adjacent_tile_index].init_transform_matrix

                        s_normal = numpy.dot(trans_source, numpy.asarray([0, 0, 1, 0]).T).T
                        t_normal = numpy.dot(trans_target, numpy.asarray([0, 0, 1, 0]).T).T
                        product = (t_normal * s_normal).sum()
                        trans = numpy.dot(numpy.linalg.inv(trans_source), trans_target)

                        trans_w_range = 1.5 / 2 * (tile_info.width_by_mm +
                                                   tile_info_dict_subsample[potential_adjacent_tile_index].width_by_mm)
                        trans_h_range = 1.5 / 2 * (tile_info.width_by_mm +
                                                   tile_info_dict_subsample[potential_adjacent_tile_index].height_by_mm)

                        if product > config["normal_direction_tolerance_by_cos"] \
                                and abs(trans[0][3]) < trans_w_range and abs(trans[1][3]) < trans_h_range:
                            tile_info.potential_loop_closure.append(potential_adjacent_tile_index)

    # save_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]), tile_info_dict_subsample)
    return tile_info_dict_subsample


def tile_info_dict_generate_kd_tree(tile_info_dict):
    tile_positions = []
    tile_index_list = []

    for tile_info_index in tile_info_dict:
        tile_info = tile_info_dict[tile_info_index]

        tile_positions.append(tile_info.position)
        # if tile_info.is_keyframe:
        #     tile_colors.append([1, 0, 0])
        #     print("Keyframe")
        # else:
        #     tile_colors.append([0, 0, 0])
        #     print("Not keyframe")
        # tile_normals.append(
        #     numpy.dot(tile_info.init_transform_matrix, numpy.asarray([0, 0, 1, 0]).T).T[0:3]
        # )
        tile_index_list.append(tile_info_index)

    tile_info_point_cloud = PointCloud()
    tile_info_point_cloud.points = Vector3dVector(tile_positions)
    tile_tree = KDTreeFlann(tile_info_point_cloud)  # Building KDtree to help searching.

    return tile_info_point_cloud, tile_tree, tile_index_list


def save_tile_info_dict(tile_info_dict_file_path, tile_info_dict):
    data_to_save = {}
    for tile_info_key in tile_info_dict:

        if tile_info_dict[tile_info_key].has_april_tag:
            april_tags_data = []
            trans_from_april_tag_data = {}
            for tag in tile_info_dict[tile_info_key].april_tags:
                tag_data = {'hamming': tag['hamming'],
                            'margin': tag["margin"],
                            'id': tag["id"],
                            'center': tag["center"].tolist(),
                            'lb-rb-rt-lt': tag["lb-rb-rt-lt"].tolist()}
                april_tags_data.append(tag_data)
            for trans_from_april_tag_id in tile_info_dict[tile_info_key].trans_from_april_tag:
                trans_from_april_tag_data[trans_from_april_tag_id] = \
                    tile_info_dict[tile_info_key].trans_from_april_tag[trans_from_april_tag_id].tolist()
        else:
            april_tags_data = []
            trans_from_april_tag_data = {}

        tile_info_to_save = {"tile_index": tile_info_dict[tile_info_key].tile_index,
                             "laplacian": tile_info_dict[tile_info_key].laplacian,
                             "is_keyframe": tile_info_dict[tile_info_key].is_keyframe,

                             "tile_info_path": tile_info_dict[tile_info_key].tile_info_path,
                             "image_path": tile_info_dict[tile_info_key].image_path,

                             "zoom_level": tile_info_dict[tile_info_key].zoom_level,

                             "position": tile_info_dict[tile_info_key].position.tolist(),
                             "rotation": tile_info_dict[tile_info_key].rotation.tolist(),

                             "width_by_pixel": tile_info_dict[tile_info_key].width_by_pixel,
                             "height_by_pixel": tile_info_dict[tile_info_key].height_by_pixel,
                             "width_by_mm": tile_info_dict[tile_info_key].width_by_mm,
                             "height_by_mm": tile_info_dict[tile_info_key].height_by_mm,

                             "has_april_tag": tile_info_dict[tile_info_key].has_april_tag,
                             "april_tags": april_tags_data,
                             "trans_from_april_tag": trans_from_april_tag_data,

                             "pose_matrix": tile_info_dict[tile_info_key].pose_matrix.tolist(),
                             "init_transform_matrix": tile_info_dict[tile_info_key].init_transform_matrix.tolist(),
                             "odometry_list": tile_info_dict[tile_info_key].odometry_list,
                             "potential_loop_closure": tile_info_dict[tile_info_key].potential_loop_closure,
                             "confirmed_loop_closure": tile_info_dict[tile_info_key].confirmed_loop_closure}
        data_to_save[tile_info_key] = tile_info_to_save
    json.dump(data_to_save, open(tile_info_dict_file_path, "w"), indent=4)


def read_tile_info_dict(tile_info_dict_file_path):
    tile_info_dict_data = json.load(open(tile_info_dict_file_path, "r"))
    tile_info_dict = {}
    for tile_info_key in tile_info_dict_data:
        new_tile_info = TileInfo(tile_index=int(tile_info_dict_data[tile_info_key]["tile_index"]),
                                 tile_info_path=tile_info_dict_data[tile_info_key]["tile_info_path"],
                                 image_path=tile_info_dict_data[tile_info_key]["image_path"],
                                 zoom_level=tile_info_dict_data[tile_info_key]["zoom_level"],
                                 position=numpy.asarray(tile_info_dict_data[tile_info_key]["position"]),
                                 rotation=numpy.asarray(tile_info_dict_data[tile_info_key]["rotation"]),
                                 width_by_mm=tile_info_dict_data[tile_info_key]["width_by_mm"],
                                 height_by_mm=tile_info_dict_data[tile_info_key]["height_by_mm"])
        new_tile_info.laplacian = float(tile_info_dict_data[tile_info_key]["laplacian"])
        new_tile_info.width_by_pixel = int(tile_info_dict_data[tile_info_key]["width_by_pixel"])
        new_tile_info.height_by_pixel = int(tile_info_dict_data[tile_info_key]["height_by_pixel"])
        new_tile_info.has_april_tag = bool(tile_info_dict_data[tile_info_key]["has_april_tag"])
        new_tile_info.is_keyframe = bool(tile_info_dict_data[tile_info_key]["is_keyframe"])

        for tag_data in tile_info_dict_data[tile_info_key]["april_tags"]:
            tag = {"hamming": int(tag_data["hamming"]),
                   "margin": float(tag_data["margin"]),
                   "id": int(tag_data["id"]),
                   "center": numpy.asarray(tag_data["center"]),
                   "lb-rb-rt-lt": numpy.asarray(tag_data["lb-rb-rt-lt"])
                   }
            new_tile_info.april_tags.append(tag)
        for trans_from_april_tag_id in tile_info_dict_data[tile_info_key]["trans_from_april_tag"]:
            new_tile_info.trans_from_april_tag[int(trans_from_april_tag_id)] = \
                numpy.asarray(tile_info_dict_data[tile_info_key]["trans_from_april_tag"][trans_from_april_tag_id])

        new_tile_info.init_transform_matrix = numpy.asarray(tile_info_dict_data[tile_info_key]["init_transform_matrix"])
        new_tile_info.pose_matrix = numpy.asarray(tile_info_dict_data[tile_info_key]["pose_matrix"])
        new_tile_info.odometry_list = tile_info_dict_data[tile_info_key]["odometry_list"]
        new_tile_info.potential_loop_closure = tile_info_dict_data[tile_info_key]["potential_loop_closure"]
        new_tile_info.confirmed_loop_closure = tile_info_dict_data[tile_info_key]["confirmed_loop_closure"]

        tile_info_dict[int(tile_info_key)] = new_tile_info

    return tile_info_dict
