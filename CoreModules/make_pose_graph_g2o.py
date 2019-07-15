import open3d
import g2o
import numpy
import cv2
from scipy.spatial.transform import Rotation
import transforms3d
import sys
sys.path.append("../Utility")
sys.path.append("../CoreFunctionModules")
from pose_estimation_cv import *
from tile_info_processing import *
from typing import Dict


class PoseGraphOptimizerG2o(g2o.SparseOptimizer):
    # Considering there would be fixed sensor data nodes, the id in pose graph is different from the real id.
    # For a certain tile with id = n, then its node id = 2n, its sensor data node id = 2n + 1.
    # But it should be transparent from outside.
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=200):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id_inside=0, trans=numpy.identity(4), fixed=False):
        # id should be the key of tiles.
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id_inside)
        v_se3.set_estimate(g2o.Isometry3d(trans))
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, s_id, t_id, local_trans=numpy.identity(4), information=numpy.identity(6), robust_kernel=None):
        # edges should be three kinds:
        # odometry: not robust, use cv_conf
        # loop closure: robust, use cv_conf
        # sensor : not robust, use fixed info.
        edge = g2o.EdgeSE3()
        edge.set_vertex(1, self.vertices()[s_id])
        edge.set_vertex(0, self.vertices()[t_id])

        edge.set_measurement(g2o.Isometry3d(local_trans))  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def add_node(self, id_outside=0, trans=numpy.identity(4), sensor_info=numpy.identity(6), fixed=False):
        id_inside = id_outside * 2
        id_sensor = id_outside * 2 + 1
        self.add_vertex(id_inside=id_inside, trans=trans, fixed=fixed)
        self.add_vertex(id_inside=id_sensor, trans=trans, fixed=True)
        self.add_edge(id_inside, id_sensor, local_trans=numpy.identity(4), information=sensor_info, robust_kernel=None)

    def add_odometry_edge(self, s_id, t_id, trans=numpy.identity(4), info=numpy.identity(6), robust_kernel=None):
        self.add_edge(s_id * 2, t_id * 2, local_trans=trans, information=info, robust_kernel=robust_kernel)

    def add_loop_closure_edge(self, s_id, t_id, trans=numpy.identity(4), info=numpy.identity(6),
                              robust_kernel=g2o.RobustKernelHuber()):
        self.add_edge(s_id * 2, t_id * 2, local_trans=trans, information=info, robust_kernel=robust_kernel)

    def get_pose(self, id_outside):
        return self.vertex(id_outside * 2).estimate().matrix()

    def get_sensor_pose(self, id_outside):
        return self.vertex(id_outside * 2 + 1).estimate().matrix()

    def make_pose_graph(self, tile_info_dict, trans_data_manager: TransDataG2o, config):
        for tile_info_key in tile_info_dict:
            tile_info = tile_info_dict[tile_info_key]
            self.add_node(id_outside=tile_info.tile_index,
                          trans=tile_info.init_transform_matrix,
                          sensor_info=trans_info_sensor_g2o(config["sensor_info_weight"]),
                          fixed=False)
        for tile_info_key in tile_info_dict:
            tile_info = tile_info_dict[tile_info_key]
            for odometry_t in tile_info.odometry_list:
                success, conf, trans = \
                    trans_data_manager.get_trans_extend(s_id=tile_info.tile_index, t_id=odometry_t)
                if success:
                    self.add_odometry_edge(s_id= tile_info.tile_index, t_id=odometry_t,
                                           trans=trans,
                                           info=trans_info_matching_g2o(conf, config["odometry_info_weight"]))

            for loop_closure_t in tile_info.potential_loop_closure:
                success, conf, trans = \
                    trans_data_manager.get_trans_extend(s_id=tile_info.tile_index, t_id=loop_closure_t)
                if success:
                    self.add_loop_closure_edge(s_id= tile_info.tile_index, t_id=loop_closure_t,
                                               trans=trans,
                                               info=trans_info_matching_g2o(conf, config["loop_closure_info_weight"]))

    def update_tile_info_dict(self, tile_info_dict):
        for tile_info_key in tile_info_dict:
            tile_info = tile_info_dict[tile_info_key]
            tile_info.pose_matrix = self.get_pose(tile_info.tile_index)
        return tile_info_dict
