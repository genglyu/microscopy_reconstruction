import os

import json
import argparse
import time, datetime
from os.path import *

import sys
sys.path.append("./Utility")
sys.path.append("./CoreModules")
sys.path.append("./Tools")
from file import *
from copy import deepcopy
from tile_info_processing import *
from visualization import *


if __name__ == "__main__":
    # set_verbosity_level(verbosity_level=VerbosityLevel.Debug)

    parser = argparse.ArgumentParser(description="Curved surface microscopy level reconstruction")

    parser.add_argument("config", help="path to the config file")

    parser.add_argument('--dict', '-d',
                        help="Options: make or load. If load, will use default path from config file")

    # estimate all the local transformation involved. Information matrix have different format.
    parser.add_argument('--register', '-r',
                        help="Options: cv_open3d, cv_g2o. Register pose graph and generate local transformations.")
    # make pose graph. Both pose graph and information matrix have different format.
    parser.add_argument('--make_pose_graph', '-m',
                        help="open3d or g2o. These two use different pose graph format.")

    parser.add_argument('--optimize', '-op',
                        help="Options: cv_open3d, cv_g2o. Optimize pose graph")

    # Visualize pose graph. Actually, raw and rough option should have same results.
    parser.add_argument('-vpg_raw',
                        action="store_true",
                        help='Visualize raw pose graph')
    parser.add_argument('-vpg_open3d_rough', '-vpg_o_r',
                        action="store_true",
                        help='Visualize rough pose graph')
    parser.add_argument('-vpg_open3d_fine', '-vpg_o_f',
                        action="store_true",
                        help='Visualize optimized pose graph')

    parser.add_argument('-vpg_g2o_rough', '-vpg_g_r',
                        action="store_true",
                        help='Visualize raw pose graph')
    parser.add_argument('-vpg_g2o_fine', '-vpg_g_f',
                        action="store_true",
                        help='Visualize optimized pose graph')
    # Visualize as point clouds or other method.
    parser.add_argument('-v_raw',
                        action="store_true",
                        help='Visualize raw data')
    parser.add_argument('-v_open3d_rough', '-v_o_r',
                        action="store_true",
                        help='Visualize rough open3d pose graph')
    parser.add_argument('-v_open3d_fine', '-v_o_f',
                        action="store_true",
                        help='Visualize optimized open3d pose graph')

    parser.add_argument('-v_g2o_rough', '-v_g_r',
                        action="store_true",
                        help='Visualize rough g2o pose graph')
    parser.add_argument('-v_g2o_fine', '-v_g_f',
                        action="store_true",
                        help='Visualize optimized g2o pose graph')

    args = parser.parse_args()

    if not args.dict\
            and not args.register\
            and not args.optimize\
            and not args.vpg_raw \
            and not args.vpg_open3d_rough \
            and not args.vpg_open3d_fine \
            and not args.vpg_g2o_rough \
            and not args.vpg_g2o_fine \
            and not args.v_raw \
            and not args.v_open3d_rough \
            and not args.v_open3d_fine \
            and not args.v_g2o_rough \
            and not args.v_g2o_fine:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.config is not None:
        with open(args.config) as json_file:
            config = json.load(json_file)
        config["path_data"] = os.path.dirname(args.config)
    assert config is not None

    # Prepare tile info dict ======================================================================
    if args.dict == "make_all":
        tile_info_dict_all = make_tile_info_dict_all(config)

    elif args.dict == "make":
        try:
            tile_info_dict_all = read_tile_info_dict(join(config["path_data"], config["tile_info_dict_all_name"]))
        except:
            tile_info_dict_all = make_tile_info_dict_all(config)
        tile_info_dict = make_info_dict(tile_info_dict_all, config)
    elif args.dict == "load":
        tile_info_dict = read_tile_info_dict(join(config["path_data"], config["tile_info_dict_name"]))
    else:
        print("Wrong input for -dict.")
        sys.exit()
    # Visualize ==================================================================================
    if args.vpg_raw:
        visualize_tile_info_dict(tile_info_dict=tile_info_dict, show_regular_frame=False)
