import numpy
import open3d
import sys

sys.path.append("../Utility")
sys.path.append("../CoreModules")
import argparse
from visualization import *
from tile_info_processing import *
import pose_estimation_cv
import cv2

# It should be the optimized tile_info_dict.

parser = argparse.ArgumentParser(description="Curved surface microscopy level reconstruction")
parser.add_argument("config", help="path to the config file")
# parser.add_argument("--tile_info_dict", "-d ", help="path to the tile info dict json file")
# parser.add_argument("--local_trans_dict", "-t ", help="path to the local trans json file")
args = parser.parse_args()

if args.config is not None:
    with open(args.config) as json_file:
        config = json.load(json_file)
    config["path_data"] = os.path.dirname(args.config)
assert config is not None

tile_info_dict = read_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]))
trans_data_manager = pose_estimation_cv.TransDataG2o(tile_info_dict, config)
trans_data_manager.read(join(config["path_data"], config["local_trans_dict_name"]))

for tile_info_key in tile_info_dict:
    tile_info = tile_info_dict[tile_info_key]

    print("amount of confirmed loops to tile %6d: %d" % (tile_info_key, len(tile_info.confirmed_loop_closure)))

    s_img = cv2.imread(tile_info.image_path)
    s_pcd = load_image_as_planar_point_cloud_open3d(image_bgr=s_img,
                                                    width_by_mm=tile_info.width_by_mm,
                                                    height_by_mm=tile_info.height_by_mm,
                                                    cv_scale_factor=-1.0,
                                                    color_filter=[2.0, 1.0, 1.0])
    s_pcd.transform(tile_info.init_transform_matrix)

    for confirmed_tile_key in tile_info.confirmed_loop_closure:
        loop_tile_info = tile_info_dict[confirmed_tile_key]
        loop_t_img = cv2.imread(loop_tile_info.image_path)
        loop_t_pcd = load_image_as_planar_point_cloud_open3d(image_bgr=loop_t_img,
                                                             width_by_mm=loop_tile_info.width_by_mm,
                                                             height_by_mm=loop_tile_info.height_by_mm,
                                                             cv_scale_factor=-1.0,
                                                             color_filter=[1.0, 1.0, 2.0])

        trans_data = trans_data_manager.get_trans(tile_info_key, confirmed_tile_key)

        trans_s_t = numpy.linalg.inv(trans_data.trans)
        trans_t = numpy.dot(tile_info.init_transform_matrix, trans_s_t)

        loop_t_pcd.transform(trans_t)

        loop_t_wire_frame = make_tile_frame(trans_matrix=loop_tile_info.init_transform_matrix,
                                            width_by_mm=loop_tile_info.width_by_mm,
                                            height_by_mm=loop_tile_info.height_by_mm,
                                            color=[0.0, 0.0, 1.0])
        s_wire_frame = make_tile_frame(trans_matrix=tile_info.init_transform_matrix,
                                       width_by_mm=tile_info.width_by_mm,
                                       height_by_mm=tile_info.height_by_mm,
                                       color=[1.0, 0.0, 0.0])



        normal_loop_init = numpy.dot(loop_tile_info.init_transform_matrix, numpy.array([1, 0, 0, 0]).T).T[0:3]
        normal_loop = numpy.dot(trans_t, numpy.array([1, 0, 0, 0]).T).T[0:3]

        print("===============================================")
        print("Confirmed loops between %6d and %6d with conf %4f" % (tile_info_key, confirmed_tile_key,
                                                                     trans_data.conf))
        # print("confidence: %4f" % trans_data.conf)
        print("normal_loop_init: ")
        print(normal_loop_init)
        print("normal_loop:")
        print(normal_loop)



        draw_geometries([s_wire_frame, s_pcd, loop_t_pcd, loop_t_wire_frame])
