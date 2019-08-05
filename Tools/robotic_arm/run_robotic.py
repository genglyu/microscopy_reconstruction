import os

import json
import argparse
import time, datetime
import file_managing
from os.path import *
from robotic_data_convert import *

import sys
sys.path.append("../../Utility")
sys.path.append("../../CoreModules")
sys.path.append("../../Tools")


if __name__ == "__main__":
    # set_verbosity_level(verbosity_level=VerbosityLevel.Debug)
    parser = argparse.ArgumentParser(description="Curved surface microscopy level reconstruction")
    parser.add_argument("robotic_config", help="path to the robotic related config file")

    parser.add_argument('--pose_list_make_all', '-d_make_all',
                        action="store_true",
                        help="Make pose_list in a single file, calculate the laplacian of images and save.")

    parser.add_argument('--align', '-a',
                        action="store_true",
                        help='Align the 3d rough reconstruction poses.')

    parser.add_argument('--interpolation', '-i',
                        action="store_true",
                        help='Interpolation through the aligned poses to generate the dense sampling points.')

    parser.add_argument('--navigation', '-n',
                        action="store_true",
                        help='Generate a relatively reasonable route through all the interpolated sampling points')

    parser.add_argument('--visualization', '-v',
                        action="store_true",
                        help='')

    args = parser.parse_args()

    if not args.pose_list_make_all\
            and not args.align\
            and not args.interpolation \
            and not args.navigation \
            and not args.visualization:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.robotic_config is not None:
        with open(args.robotic_config) as json_file:
            robotic_config = json.load(json_file)
        robotic_config["path_data"] = os.path.dirname(args.robotic_config)
    assert robotic_config is not None

    # Prepare tile info dict ======================================================================
    if args.pose_list_make_all:
        file_managing.touch_folder(join(robotic_config["path_data"],
                                        robotic_config["robotic_reconstruction_workspace"]))
        pose_list, trans_list = \
            make_robotic_pose_list(join(robotic_config["path_data"],
                                        robotic_config["robotic_reconstruction_pose_dir"]))
        save_robotic_pose_list(join(robotic_config["path_data"],
                                    robotic_config["robotic_reconstruction_workspace"],
                                    robotic_config["robotic_reconstruction_pose_list_all"]),
                               pose_list)
        save_trans_list(join(robotic_config["path_data"],
                             robotic_config["robotic_reconstruction_workspace"],
                             robotic_config["robotic_reconstruction_trans_list_all"]),
                        trans_list)
    if args.align:
        import pose_align
        trans_list = read_trans_list(join(robotic_config["path_data"],
                                          robotic_config["robotic_reconstruction_workspace"],
                                          robotic_config["robotic_reconstruction_trans_list_all"]))
        optimizer = pose_align.PoseGraphOptimizerG2oRobotic()
        optimizer.load_trans_list(trans_list)
        optimizer.save_pose_graph(join(robotic_config["path_data"],
                                       robotic_config["robotic_reconstruction_workspace"],
                                       robotic_config["original_pose_graph"]))

        optimizer.optimize()

        optimizer.save_pose_graph(join(robotic_config["path_data"],
                                       robotic_config["robotic_reconstruction_workspace"],
                                       robotic_config["aligned_pose_graph"]))

        trans_list_aligned = optimizer.export_optimized_as_trans_list()
        save_trans_list(join(robotic_config["path_data"],
                             robotic_config["robotic_reconstruction_workspace"],
                             robotic_config["robotic_reconstruction_trans_list_aligned"]))


    if args.interpolation:
        import robotic_surface_interpolation
        trans_list_aligned = read_trans_list(join(robotic_config["path_data"],
                                                  robotic_config["robotic_reconstruction_workspace"],
                                                  robotic_config["robotic_reconstruction_trans_list_aligned"]))
        surface_interpolator = robotic_surface_interpolation.RoboticSurfaceConstructor()

        surface_interpolator.load_trans_list(trans_list_aligned)
        surface_interpolator.run_interpolation()
        surface_interpolator.save_interpolated_robotic_pose(
            join(robotic_config["path_data"],
                 robotic_config["robotic_reconstruction_workspace"],
                 robotic_config["robotic_reconstruction_trans_interpolated"]))

    if args.navigation:
        import navigation
        trans_list_interpolated = read_trans_list(join(robotic_config["path_data"],
                                                       robotic_config["robotic_reconstruction_workspace"],
                                                       robotic_config["robotic_reconstruction_trans_interpolated"]))

        navigator = navigation.NavigationGraph()

        points = trans_list_to_points(trans_list_interpolated)
        navigator.load_point_list(points, 0.03)
        order = navigator.dfs()
        # order = navigator.bfs(2)

        # points = adjust_order(points, order)
        trans_list_ordered = adjust_order(trans_list_interpolated, order)
        save_trans_list(join(robotic_config["path_data"],
                             robotic_config["robotic_reconstruction_workspace"],
                             robotic_config["robotic_reconstruction_trans_ordered"]),
                        trans_list_ordered)

        save_robotic_full_pose_list(join(robotic_config["path_data"],
                                         robotic_config["robotic_reconstruction_workspace"],
                                         robotic_config["robotic_reconstruction_trans_ordered"]),
                                    trans_list_to_pos_ori_list(trans_list_ordered))

    if args.visualization:
        import robotic_visualizer
        viewer = robotic_visualizer.RoboticVisualizerOpen3d(robotic_config)
        viewer.view_via_robotic_config()

    # Visualize ==================================================================================


