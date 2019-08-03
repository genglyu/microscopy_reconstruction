import g2o
import numpy
from robotic_data_convert import *
from open3d import *


# It should be reasonable to say no more than 400000 nodes can be fed into this... Even considering the weird index.
class PoseGraphOptimizerG2oRobotic(g2o.SparseOptimizer):
    # Considering there would be fixed sensor data nodes, the id in pose graph is different from the real id.
    # For a certain tile with id = n, then its node id = 2n, its sensor data node id = 2n + 1.
    # But it should be transparent from outside.
    def __init__(self):
        self.sensor_id_offset = 400000
        self.align_radius = 0.005
        self.node_amount = 0
        super().__init__()
        # solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        # algorithm = g2o.OptimizationAlgorithmDogleg(solver)

        # Show logs to see it is running.
        super().set_verbose(True)
        super().set_algorithm(algorithm)

    def optimize(self, max_iterations=500):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_robotic_pose_node(self, id_outside=0,
                              trans=numpy.identity(4)):
        id_inside = id_outside
        id_init_pose = id_outside + self.sensor_id_offset

        self.add_vertex(id_inside=id_inside, trans=trans, fixed=False)
        self.add_vertex(id_inside=id_init_pose, trans=trans, fixed=True)
        self.add_edge(id_inside, id_init_pose,
                      local_trans=numpy.identity(4),
                      information=numpy.array([[100, 0, 0, 0, 0, 0],
                                               [0, 100, 0, 0, 0, 0],
                                               [0, 0, 100, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 0, 1]]),
                      robust_kernel=None)

    def add_neighbour_edge(self, s_id, t_id, distance=0.0):
        weight = 60 * self.align_radius / (distance + self.align_radius / 10)
        angle_weight = 0
        print("Weight: %5f" % weight)
        self.add_edge(s_id, t_id,
                      local_trans=numpy.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]),
                      information=numpy.array([[0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [0, 0, weight, 0, 0, 0],
                                               [0, 0, 0, angle_weight, 0, 0],
                                               [0, 0, 0, 0, angle_weight, 0],
                                               [0, 0, 0, 0, 0, 0]]),
                      robust_kernel=None)

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

    def get_pose(self, id_outside):
        return self.vertex(id_outside).estimate().matrix()

    def load_trans_list(self, trans_list):
        points = []
        self.node_amount = len(trans_list)
        for i, trans in enumerate(trans_list):
            points.append(numpy.dot(trans, numpy.array([0, 0, 0, 1]).T).T[0:3].tolist())
            self.add_robotic_pose_node(i, trans)
        pcd = PointCloud()
        pcd.points = Vector3dVector(points)
        kd_tree = KDTreeFlann(pcd)

        for i, point in enumerate(points):
            [_, idx, _] = kd_tree.search_radius_vector_3d(point, radius=self.align_radius)
            for id in idx:
                if id > i:
                    distance = numpy.linalg.norm(numpy.asarray(points[i]) - numpy.asarray(points[id]))
                    self.add_neighbour_edge(i, id, distance)

    def export_optimized_as_trans_list(self):
        trans_list = []
        for i in range(self.node_amount):
            trans_list.append(self.get_pose(i))
        return trans_list

    def export_optimised_as_robotic_pose(self):
        robotic_pose_list = []
        for i in range(self.node_amount):
            robo_pose = trans_to_rob_pose(self.get_pose(i))
            robotic_pose_list.append(robo_pose)
        return robotic_pose_list
