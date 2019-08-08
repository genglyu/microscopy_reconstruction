from open3d import *
import robotic_data_convert
import numpy
import math
from scipy.spatial.transform import Rotation
import transforms3d


def rotation_matrix(normal):
    a = numpy.asarray(normal)
    [x, y, z] = (a / numpy.linalg.norm(a)).tolist()
    projected_yz = y*y + z*z
    if projected_yz != 0:
        rot_y = math.acos(x)
        rot_x = math.asin(y / math.sqrt(projected_yz))
        if z < 0:
            rot_x = math.pi - rot_x
        [n_x, n_y, n_z] = Rotation.from_euler("yxz", [-rot_y, -rot_x, 0]).as_euler("xyz")
        rotation = Rotation.from_euler("xyz", [0, n_y, n_z]).as_dcm()
    else:
        if x >= 0:
            rotation = numpy.identity(3)
        else:
            rotation = Rotation.from_euler("xyz", [0, math.pi, 0]).as_dcm()
    return rotation


point_list = robotic_data_convert.read_points_list("/home/lvgeng/Code/TestingData/robotic/matlab/matlab.json")
pcd = PointCloud()
pcd.points = Vector3dVector(point_list)
estimate_normals(pcd)

normals = []
for normal in pcd.normals:
    if normal[2] < 0:
        normals.append(normal * -1)
    else:
        normals.append(normal)
pcd.normals = Vector3dVector(normals)
pcd_down = geometry.voxel_down_sample(pcd, voxel_size=0.001)


trans_list = []
for i, position in enumerate(pcd_down.points):
    rotation_m = rotation_matrix(pcd_down.normals[i])
    trans = transforms3d.affines.compose(T=position, R=rotation_m, Z=[1, 1, 1])
    trans_list.append(trans)

robotic_data_convert.save_trans_list(
    path="/home/lvgeng/Code/TestingData/robotic/matlab/robotic_reconstruction_trans_interpolated.json",
    trans_list=trans_list)


draw_geometries([pcd_down])
