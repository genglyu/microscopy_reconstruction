import numpy
import os
import sys
import json
import argparse
import time, datetime
from os.path import *
from scipy.spatial.transform import Rotation
import transforms3d

sys.path.append("./Utility")
sys.path.append("./CoreFunctionModules")
sys.path.append("./EvaluationAndCalibration")
from pose_estimation_cv_icp import *
from file_managing import *
sys.path.append(".")
from initialize_config import *
from tile_info_preprocessing import *
from copy import deepcopy
import multiprocessing
from joblib import Parallel, delayed
from visualization_results import *


# Designed for camera calibration only.



def build_equations_list(trans_cv, trans_sensor):
    v=trans_cv
    s=trans_sensor
    # coefficient = [
    #   # [a,                 b,                 c,                 d,                 e,                 f,                 g,                 h,                 i,                 j,                 k,                 l,                 1]
    #     [v[0][0]-s[0][0],   v[1][0],           v[2][0],           v[3][0],           -s[0][1],          0,                 0,                 0,                 -s[0][2],          0,                 0,                 0               ],
    #     [v[0][1],           v[1][1]-s[0][0],   v[2][1],           v[3][1],           0,                 -s[0][1],          0,                 0,                 0,                 -s[0][2],          0,                 0               ],
    #     [v[0][2],           v[1][2],           -s[0][0],          v[3][2],           0,                 0,                 -s[0][1],          0,                 0,                 0,                 -s[0][2],          0               ],
    #     [v[0][3],           v[1][3],           v[2][3],           v[3][3]-s[0][0],   0,                 0,                 0,                 -s[0][1],          0,                 0,                 -s[0][2],          0               ],
    #     [-s[1][0],          0,                 0,                 0,                 v[0][0]-s[1][1],   v[1][0],           v[2][0],           v[3][0],           -s[1][2],          0,                 0,                 0               ],
    #     [0,                 -s[1][0],          0,                 0,                 v[0][1],           v[1][1]-s[1][1],   v[2][1],           v[3][1],           0,                 -s[1][2],          0,                 0               ],
    #     [0,                 0,                 -s[1][0],          0,                 v[0][2],           v[1][2],           v[2][2]-s[1][1],   v[3][2],           0,                 0,                 -s[1][2],          0               ],
    #     [0,                 0,                 0,                 -s[1][0],          v[0][3],           v[1][3],           v[2][3],           v[3][3]-s[1][1],   0,                 0,                 0,                 -s[1][2]        ],
    #     [-s[2][0],          0,                 0,                 0,                 -s[2][1],          0,                 0,                 0,                 v[0][0]-s[2][2],   v[1][0],           v[2][0],           v[3][0]         ],
    #     [0,                 -s[2][0],          0,                 0,                 0,                 -s[2][1],          0,                 0,                 v[0][1],           v[1][1]-s[2][2],   v[2][1],           v[3][1]         ],
    #     [0,                 0,                 -s[2][0],          0,                 0,                 0,                 -s[2][1],          0,                 v[0][2],           v[1][2],           v[2][2]-s[2][2],   v[3][2]         ],
    #     [0,                 0,                 0,                 -s[2][0],          0,                 0,                 0,                 -s[2][1],          v[0][3],           v[1][3],           v[2][3],           v[3][3]-s[2][2] ],
    #     [-s[3][0],          0,                 0,                 0,                 -s[3][1],          0,                 0,                 0,                 -s[3][2],          0,                 0,                 0               ],
    #     [0,                 -s[3][0],          0,                 0,                 0,                 -s[3][1],          0,                 0,                 0,                 -s[3][2],          0,                 0               ],
    #     [0,                 0,                 -s[3][0],          0,                 0,                 0,                 -s[3][1],          0,                 0,                 0,                 -s[3][2],          0               ],
    #     [0,                 0,                 0,                 -s[3][0],          0,                 0,                 0,                 -s[3][1],          0,                 0,                 0,                 -s[3][2]        ]
    # ]
    coefficient = [
      # [a,                 b,                 c,                 d,                 e,                 f,                 g,                 h,                 i,                 j,                 k,                 l,                 1]
        [v[0][0]-s[0][0],   v[1][0],           v[2][0],           v[3][0],           -s[0][1],          0,                 0,                 0,                 -s[0][2],          0,                 0,                 0               ],
        [v[0][1],           v[1][1]-s[0][0],   v[2][1],           v[3][1],           0,                 -s[0][1],          0,                 0,                 0,                 -s[0][2],          0,                 0               ],
        [v[0][2],           v[1][2],           -s[0][0],          v[3][2],           0,                 0,                 -s[0][1],          0,                 0,                 0,                 -s[0][2],          0               ],
        [v[0][3],           v[1][3],           v[2][3],           v[3][3]-s[0][0],   0,                 0,                 0,                 -s[0][1],          0,                 0,                 -s[0][2],          0               ],
        [-s[1][0],          0,                 0,                 0,                 v[0][0]-s[1][1],   v[1][0],           v[2][0],           v[3][0],           -s[1][2],          0,                 0,                 0               ],
        [0,                 -s[1][0],          0,                 0,                 v[0][1],           v[1][1]-s[1][1],   v[2][1],           v[3][1],           0,                 -s[1][2],          0,                 0               ],
        [0,                 0,                 -s[1][0],          0,                 v[0][2],           v[1][2],           v[2][2]-s[1][1],   v[3][2],           0,                 0,                 -s[1][2],          0               ],
        [0,                 0,                 0,                 -s[1][0],          v[0][3],           v[1][3],           v[2][3],           v[3][3]-s[1][1],   0,                 0,                 0,                 0        ],
        [0,          0,                 0,                 0,                 -s[2][1],          0,                 0,                 0,                 v[0][0]-s[2][2],   v[1][0],           0,           0         ],
        [0,                 0,          0,                 0,                 0,                 -s[2][1],          0,                 0,                 v[0][1],           v[1][1]-s[2][2],   v[2][1],           0         ],
        [0,                 0,                 0,          0,                 0,                 0,                 -s[2][1],          0,                 v[0][2],           v[1][2],           v[2][2]-s[2][2],   0         ],
        [0,                 0,                 0,                 0,          0,                 0,                 0,                 -s[2][1],          v[0][3],           v[1][3],           v[2][3],           0 ],
        [-s[3][0],          0,                 0,                 0,                 -s[3][1],          0,                 0,                 0,                 -s[3][2],          0,                 0,                 0               ],
        [0,                 -s[3][0],          0,                 0,                 0,                 -s[3][1],          0,                 0,                 0,                 -s[3][2],          0,                 0               ],
        [0,                 0,                 -s[3][0],          0,                 0,                 0,                 -s[3][1],          0,                 0,                 0,                 -s[3][2],          0               ],
        [0,                 0,                 0,                 -s[3][0],          0,                 0,                 0,                 -s[3][1],          0,                 0,                 0,                 0        ]
    ]

    dependent_v = [
        0,
        0,
        0,
        s[0][3],
        0,
        0,
        0,
        s[1][3],
        0,
        0,
        0,
        s[2][3],
        -v[3][0],
        -v[3][1],
        -v[3][2],
        s[3][3]-v[3][3]
    ]

    # dependent_v = [
    #     [0],
    #     [0],
    #     [0],
    #     [s[0][3]],
    #     [0],
    #     [0],
    #     [0],
    #     [s[1][3]],
    #     [0],
    #     [0],
    #     [0],
    #     [s[2][3]],
    #     [-v[3][0]],
    #     [-v[3][1]],
    #     [-v[3][2]],
    #     [s[3][3]-v[3][3]]
    # ]
    return coefficient, dependent_v


def calculate_trans(reference_tile_info, tile_info, config):
    (success, trans_cv, conf) = transform_estimation_opencv(tile_info, reference_tile_info, config)
    if success:
        # trans_sensor = numpy.dot(tile_info.init_transform_matrix,
        #                          numpy.linalg.inv(reference_tile_info.init_transform_matrix))
        trans_sensor = numpy.dot(
            tile_info.init_transform_matrix, numpy.linalg.inv(reference_tile_info.init_transform_matrix))

        (coefficient, dependent_v) = build_equations_list(trans_cv, trans_sensor)
        return True, coefficient, dependent_v
    else:
        return False, numpy.zeros((16, 12)).tolist(), numpy.zeros(16).tolist()


def parallel_matching(tile_info_dict, config):
    max_thread = min(multiprocessing.cpu_count(), max(len(tile_info_dict), 1))

    matching_results = Parallel(n_jobs=max_thread)(
        delayed(calculate_trans)(
            tile_info_dict[0],
            tile_info_dict[tile_info_key],
            config
        )
        for tile_info_key in tile_info_dict)

    coefficient = []
    dependent_v = []

    for result in matching_results:
        print(result)
        if result[0]:
            coefficient += result[1]
            dependent_v += result[2]

    C, residules, rank, singval = numpy.linalg.lstsq(coefficient, dependent_v)
    print(C)
    camera_offset = numpy.asarray(C.tolist()+[0.0,0.0,0.0,1.0]).reshape((4, 4))

    (ct, cr, cz, cs) = transforms3d.affines.decompose44(camera_offset)
    euler_cv = Rotation.from_dcm(cr).as_euler("xyz")

    print(camera_offset)
    print(ct)
    print(euler_cv)
    print(cz)
    print(cs)


    return camera_offset

# def calculate_trans(reference_tile_info, tile_info, config):
#     # (success, trans_cv, conf) = transform_estimation_opencv(tile_info, reference_tile_info, config)
#     (success, trans_cv, conf) = transform_estimation_opencv(tile_info, reference_tile_info, config)
#
#     if success:
#         trans_sensor = numpy.dot(
#             tile_info.init_transform_matrix, numpy.linalg.inv(reference_tile_info.init_transform_matrix))
#
#         trans_offset = numpy.dot(numpy.linalg.inv(trans_sensor), trans_cv)
#         # trans_offset = numpy.dot(trans_cv, numpy.linalg.inv(trans_sensor))
#         # trans_offset = numpy.dot(numpy.linalg.inv(trans_cv), trans_sensor)
#         # trans_offset = numpy.dot(trans_cv, numpy.linalg.inv(trans_sensor))
#
#
#         (ct, cr, cz, cs) = transforms3d.affines.decompose44(trans_cv)
#         euler_cv = Rotation.from_dcm(cr).as_euler("xyz")
#         (pt, pr, pz, ps) = transforms3d.affines.decompose44(trans_offset)
#         euler = Rotation.from_dcm(pr).as_euler("xyz")
#
#         return True, numpy.asarray(pt), numpy.asarray(euler), numpy.asarray(pz), numpy.asarray(euler_cv)
#     else:
#         return False, numpy.asarray([0.0,0.0,0.0]), numpy.asarray([0.0, 0.0, 0.0]), numpy.asarray([1.0, 1.0, 1.0]), \
#                numpy.asarray([0.0, 0.0, 0.0])
#
#
# def parallel_matching(tile_info_dict, config):
#     max_thread = min(multiprocessing.cpu_count(), max(len(tile_info_dict), 1))
#     matching_results = Parallel(n_jobs=max_thread)(
#         delayed(calculate_trans)(
#             tile_info_dict[0],
#             tile_info_dict[tile_info_key],
#             config
#         )  # Can be optimised by loading only once
#         for tile_info_key in tile_info_dict)
#
#
#     pt = numpy.asarray([0.0, 0.0, 0.0])
#     euler = numpy.asarray([0.0, 0.0, 0.0])
#     pz = numpy.asarray([0.0, 0.0, 0.0])
#
#     for result in matching_results:
#         print(result[0],result[1], result[2], result[4])
#         if result[0]:
#             pt += result[1]
#             euler += result[2]
#             pz += result[3]
#     pt = pt / len(matching_results)
#     rotation = Rotation.from_euler("xyz", euler / len(matching_results)).as_dcm()
#     pz = pz / len(matching_results)
#
#     camera_offset = transforms3d.affines.compose(pt, rotation, pz)
#     print(camera_offset)
#
#     return camera_offset






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("--make",
                        help="Make tile list and generate potential edge connections", action="store_true")
    parser.add_argument("--load",
                        help="Load tile list and generate potential edge connections", action="store_true")
    args = parser.parse_args()
    if not args.make and \
            not args.load:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.config is not None:
        with open(args.config) as json_file:
            config = json.load(json_file)
            initialize_config(config)
            config["path_data"] = os.path.dirname(args.config)
    assert config is not None

    if args.load:
        tile_info_dict = read_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]))
    elif args.make:
        tile_info_file_list = get_file_list(join(config["path_data"], config["path_tile_info"]))
        tile_info_dict = make_tile_info_dict(tile_info_file_list,
                                             join(config["path_data"], config["path_image_dir"]),
                                             config["index_start"], config["index_end"],
                                             config["subsample_per_n_frame"])
        reference_pos = deepcopy(tile_info_dict[0].position)
        reference_rot = deepcopy(tile_info_dict[0].rotation)

        tile_info_dict = tile_info_dict_recenter(tile_info_dict, reference_pos, reference_rot)
        # tile_info_dict = tile_info_dict_recenter_deep(tile_info_dict, reference_pos, reference_rot)

        (tile_info_pcd, tile_info_tree, tile_index_list) = tile_info_dict_generate_cloud_and_kd_tree(tile_info_dict)
        tile_info_dict = tile_info_dict_generate_potential_loop_closure(tile_info_dict, tile_info_tree, tile_index_list,
                                                                        config["n_keyframes_per_n_frame"])

        potential_bug_in_dict(tile_info_dict)

        save_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]), tile_info_dict)
    else:
        print("No tile info dict.")
    touch_folder(join(config["path_data"], config["path_local_cv_transformation"]))
    print("Start estimating camera offset.")

    camera_offset = parallel_matching(tile_info_dict, config)
    camera_offset_inv = numpy.linalg.inv(camera_offset)

    for tile_info_key in tile_info_dict:
        tile_info_dict[tile_info_key].init_transform_matrix = numpy.dot(camera_offset_inv, numpy.dot(tile_info_dict[tile_info_key].init_transform_matrix, camera_offset))

    visualize_tiles_as_point_cloud(tile_info_dict, config["visualization_voxel_size"])



