from open3d import *
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
from tile_info_processing import *


























# preparing data
original_pose = numpy.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 2],
                             [0, 0, 0, 1]])
pose_list = []
pcd_all = PointCloud()

tile_info_dict = {}

rotations_eul = numpy.random.randint(0, 200, size=(400, 3)) / 100.0
# print(rotations_eul)
#
for i, eu in enumerate(rotations_eul):
    pose = numpy.dot(transforms3d.affines.compose([0, 0, 0], Rotation.from_euler("xyz", eu).as_dcm(), [1, 1, 1]),
                     numpy.asarray(original_pose))

    pose_list.append(pose)
    pcd = PointCloud()
    pcd.points = Vector3dVector(numpy.asarray([[0, 0, 0]]))
    pcd.colors = Vector3dVector(numpy.asarray([[0, 1, 0]]))
    pcd.normals = Vector3dVector(numpy.asarray([[0, 0, 1]]))
    pcd.transform(pose)
    pcd_all += pcd

    tile_info_dict[i] = TileInfo(tile_index=i,
                                 tile_info_path="",
                                 image_path="",
                                 zoom_level=0,
                                 position=[0, 0, 0], rotation=[0, 0, 1, 0])
    tile_info_dict[i].pose_matrix = pose
    tile_info_dict[i].position = pcd.points[0]

# draw_geometries([pcd_all])

# try to build the surface
_, tile_tree, tile_index_list = tile_info_dict_generate_kd_tree(tile_info_dict)


def interpolating_for_tile(id, radius, tile_info_dict, tile_tree, tile_index_list):
    tile_info = tile_info_dict[id]
    [_, idx, _] = tile_tree.search_radius_vector_3d(tile_info.position, radius=radius)
    normalized_positions = []
    for index in idx:
        key = tile_index_list[index]
        # in_range_key_list.append(key)
        posistion = numpy.dot(numpy.linalg.inv(tile_info.pose_matrix),
                              numpy.dot(tile_info_dict[key].pose_matrix,
                                        numpy.array([0, 0, 0, 1]).T)).T[0:3]
        normalized_positions.append(posistion)

    # print(normalized_positions)

    normalized_positions = numpy.asarray(normalized_positions)
    # print(normalized_positions)
    points = normalized_positions[:,0:2]
    values = normalized_positions[:,2:3]
    # print(points)
    # print(values)
    grid_x, grid_y = np.mgrid[-2:2:100j, -2:2:100j]
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    print(numpy.asarray(grid_x).reshape((1,-1)))
    pos_list = numpy.asarray(
        [numpy.asarray(grid_x).reshape((1,-1))[0],
        numpy.asarray(grid_y).reshape((1,-1))[0],
        numpy.asarray(grid_z).reshape((1,-1))[0]]).T
    # print(type(pos_list[0][2]))
    # if math.isnan(pos_list[0][2]):
    #     print("is nan")
    points = pos_list[numpy.logical_not(numpy.isnan(pos_list[:, 2]))]
    print(len(pos_list))
    print(points)

    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    pcd.colors = Vector3dVector(numpy.repeat([[1, 0, 0]], len(normalized_positions), axis=0))

    geometry.estimate_normals(pcd, search_param=geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

    normal_list = []
    for normal in pcd.normals:
        if normal[2] < 0:
            normal_list.append(normal * -1)
        else:
            normal_list.append(normal)
    pcd.normals = Vector3dVector(normal_list)

    return pcd



# interpolating_for_tile(0, 1, tile_info_dict, tile_tree, tile_index_list)

id = 5
radius = 2
pcd_in_range = interpolating_for_tile(id, radius, tile_info_dict, tile_tree, tile_index_list)
pcd_in_range.transform(numpy.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, -0.01],
                                    [0, 0, 0, 1]]))
pcd_in_range.transform(tile_info_dict[id].pose_matrix)

pcd_center = PointCloud()
print(tile_info_dict[id].position)
pcd_center.points = Vector3dVector([tile_info_dict[id].position])
pcd_center.colors = Vector3dVector([[0, 0, 1]])

# all = pcd_all+pcd_in_range+pcd_center
# geometry.estimate_normals(all, search_param=geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

# draw_geometries([all])

draw_geometries([pcd_all, pcd_in_range, pcd_center])