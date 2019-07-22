import open3d
import numpy
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import transforms3d

import math

import sys
sys.path.append("../../Utility")
sys.path.append("../../CoreModules")
sys.path.append("../../Tools")


# pipeline:
# 1. use all the transformations to make a point cloud and its kd tree.


# The create the full surface of the entire scanned area.
class SurfaceConstructor:
    def __init__(self):
        self.pose_list = []
        self.points_list = []
        self.sampling_pcd = None
        self.sampling_kd_tree = None
        self.interpolated_face_list = []
        self.integrated_surface_pcd = []
    # Find the closest actually sampled point and then add rotation to the transformation.


# The Interpolator used to create the area around a certain sampling point
class SurfaceInterpolator:
    def __init__(self):
        self.pose_list = []
        self.points_list = []

        self.referenced_xy = numpy.array([[0, 0]])
        self.referenced_z = numpy.array([[0]])

    def get_interpolated_value(self, x, y):
        # grid_x, grid_y = np.mgrid[-2:2:100j, -2:2:100j]
        interpolated_z = griddata(self.referenced_xy, self.referenced_z, (x, y), method='cubic')
        return interpolated_z


def rotation_matrix(normal_s, normal_t):
    a = normal_s / numpy.linalg.norm(normal_s)
    b = normal_t / numpy.linalg.norm(normal_t)
    v = numpy.cross(a, b)
    vx = numpy.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
    rotation = numpy.identity(3) + vx + numpy.dot(vx, vx) * 1 / (1 + numpy.linalg.norm(a * b))
    euler = Rotation.from_dcm(rotation).as_euler("xyz")
    # In this case, it assumes that the referenced normal direction is the z direction.
    # It would be better if using x direction cause the euler rotations are applied by order x, y, z.
    # The rotation along normal direction is related to image feature matching (z direction in this case.)
    # However, since it is the last one having effect, it also changes the x y rotation values.
    r_z = Rotation.from_euler("xyz", [0, 0, -euler[2]]).as_dcm()
    rotation = numpy.dot(rotation, r_z)
    return rotation

