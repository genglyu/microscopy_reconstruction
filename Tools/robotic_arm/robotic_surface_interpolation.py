import open3d
import numpy
from scipy.interpolate import griddata
from scipy.interpolate import interp2d

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import transforms3d
import math

import sys
sys.path.append("../../Utility")
sys.path.append("../../CoreModules")
sys.path.append("../../Tools")
from tile_info_processing import *
import visualization
import g2o
from robotic_data_convert import *

trans_rob_to_camera = transforms3d.affines.compose([0, 0, 0], Rotation.from_euler("xyz", [0, 0, 0]).as_dcm(), [1, 1, 1])
trans_camera_to_rob = numpy.linalg.inv(trans_rob_to_camera)


def rob_pose_to_trans(rob_pose):
    return numpy.dot(numpy.asarray(rob_pose).reshape((4, 4)).T, trans_rob_to_camera)


def trans_to_rob_pose(trans):
    return numpy.dot(numpy.asarray(trans), trans_camera_to_rob).T.reshape((-1)).tolist()


class RoboticSurfaceConstructor:
    def __init__(self):
        self.trans_list = []
        self.points_list = []
        self.normal_list = []
        self.sampling_pcd = PointCloud()
        self.sampling_kd_tree = None

        self.interpolated_trans_list = []
        self.interpolate_robotic_pose = []
        self.integrated_surface_pcd = PointCloud()
    # Find the closest actually sampled point and then add rotation to the transformation.

    def load_trans_list(self, trans_list):
        for i, trans in enumerate(trans_list):
            self.trans_list.append(trans)
            self.points_list.append(numpy.dot(trans, numpy.array([0, 0, 0, 1]).T).T[0:3].tolist())
            self.normal_list.append(numpy.dot(trans, numpy.array([1, 0, 0, 0]).T).T[0:3].tolist())

        self.points_list = numpy.asarray(self.points_list)
        self.sampling_pcd.points = Vector3dVector(numpy.asarray(self.points_list))
        self.sampling_pcd.normals = Vector3dVector(numpy.asarray(self.normal_list))
        self.sampling_pcd.colors = Vector3dVector(numpy.repeat(numpy.array([[0, 0, 0]]), len(self.points_list), axis=0))

        self.sampling_kd_tree = KDTreeFlann(self.sampling_pcd)
        # draw_geometries([self.sampling_pcd])

    def interpolate_sub(self, index=0, radius=0.008, interpolate_w=0.01, interpolate_h=0.01, interpolate_amount=100):

        [_, idx, _] = self.sampling_kd_tree.search_radius_vector_3d(self.points_list[index], radius=radius)

        idx = list(idx)
        print("interpolate for %d from %d in range points" % (index, len(idx)))

        sub_points = numpy.c_[self.points_list[idx, :], numpy.ones(len(idx))]

        transformed_sub_points = numpy.dot(numpy.linalg.inv(self.trans_list[index]), sub_points.T).T[:, 0:3]

        yz = transformed_sub_points[:, 1:3]

        x = transformed_sub_points[:, 0]
        y = transformed_sub_points[:, 1]
        z = transformed_sub_points[:, 2]

        # print(yz)

        grid_y, grid_z = numpy.mgrid[
                         -interpolate_w/2:interpolate_w/2:(interpolate_amount * 1j),
                         -interpolate_h/2:interpolate_h/2:(interpolate_amount * 1j)]
        grid_x = griddata(yz, x, (grid_y, grid_z), method='cubic')
        interpolate_points = numpy.c_[grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)]
        interpolate_points = interpolate_points[numpy.logical_not(numpy.isnan(interpolate_points[:, 0]))]

        sub_pcd = PointCloud()
        sub_pcd.points = Vector3dVector(interpolate_points)

        geometry.estimate_normals(sub_pcd,
                                  search_param=geometry.KDTreeSearchParamHybrid(radius=interpolate_w, max_nn=30))
        sub_pcd.normalize_normals()

        normal_list = []
        for normal in sub_pcd.normals:
            if normal[0] < 0:
                normal_list.append(normal * -1)
            else:
                normal_list.append(normal)
        sub_pcd.normals = Vector3dVector(normal_list)

        sub_pcd.transform(self.trans_list[index])
        # draw_geometries([sub_pcd, self.sampling_pcd])
        return sub_pcd

    def run_interpolation(self):
        # Down sample the entire point cloud to pick the ones that do interpolation on ================================
        draw_geometries([self.sampling_pcd])
        cl, ind = open3d.geometry.statistical_outlier_removal(self.sampling_pcd,
                                                              nb_neighbors=300,
                                                              std_ratio=0.01)
        display_inlier_outlier(self.sampling_pcd, ind)
        # interpolating_point_trans_list = adjust_order(self.trans_list, ind)

        for i in ind:
            sub_pcd = self.interpolate_sub(index=i)
            # sub_pcd.transform(pose)
            self.integrated_surface_pcd = self.integrated_surface_pcd + sub_pcd

        # print(self.integrated_surface_pcd.normals[100])

        # ============================================================
        # self.integrated_surface_pcd = geometry.voxel_down_sample(self.integrated_surface_pcd, voxel_size=0.001)
        # self.integrated_surface_pcd = open3d.geometry.uniform_down_sample(input=self.integrated_surface_pcd,
        #                                                                   every_k_points=100)

        # draw_geometries([self.integrated_surface_pcd])

        min_cube_size = 0.002
        pcd_down = geometry.voxel_down_sample(input=self.integrated_surface_pcd,
                                              voxel_size=min_cube_size)

        draw_geometries([pcd_down])

        min_bound = pcd_down.get_min_bound() - min_cube_size * 0.5
        max_bound = pcd_down.get_max_bound() + min_cube_size * 0.5

        pcd_down, index_in_pcd = \
            geometry.voxel_down_sample_and_trace(input=self.integrated_surface_pcd,
                                                 voxel_size=min_cube_size,
                                                 min_bound=min_bound,
                                                 max_bound=max_bound,
                                                 approximate_class=False)
        # self.integrated_surface_pcd = pcd_down
        # print(index_in_pcd)
        index_list = index_in_pcd.reshape(-1)
        index_list = numpy.sort(index_list[index_list >= 0]).tolist()

        points = []
        normals = []
        # colors = []
        for index in index_list:
            points.append(self.integrated_surface_pcd.points[index])
            normals.append(self.integrated_surface_pcd.normals[index])
            # colors.append(self.integrated_surface_pcd.colors[index])
        self.integrated_surface_pcd = PointCloud()
        self.integrated_surface_pcd.points = Vector3dVector(points)
        # self.integrated_surface_pcd = self.integrated_surface_pcd + self.sampling_pcd

        self.integrated_surface_pcd.normals = Vector3dVector(normals)
        # self.integrated_surface_pcd.colors = Vector3dVector(colors)

        # geometry.estimate_normals(self.integrated_surface_pcd,
        #                           search_param=geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

        # cl, ind = open3d.geometry.radius_outlier_removal(self.integrated_surface_pcd,
        #                                                  nb_points=5,
        #                                                  radius=0.006)
        cl, ind = open3d.geometry.statistical_outlier_removal(self.integrated_surface_pcd,
                                                              nb_neighbors=10,
                                                              std_ratio=1.0)
        display_inlier_outlier(self.integrated_surface_pcd, ind)

        # draw_geometries([self.integrated_surface_pcd])
        # ============================================================
        # connection = visualization.make_connection_of_pcd_order(self.integrated_surface_pcd)
        # draw_geometries([self.integrated_surface_pcd, connection])
        # ============================================================

        for i, position in enumerate(self.integrated_surface_pcd.points):
            rotation_m = rotation_matrix(self.integrated_surface_pcd.normals[i])

            trans = transforms3d.affines.compose(T=position, R=rotation_m, Z=[1, 1, 1])
            self.interpolated_trans_list.append(trans.tolist())
            self.interpolate_robotic_pose.append(trans_to_rob_pose(trans))






    def save_interpolated_robotic_pose(self, save_path):
        json.dump(self.interpolate_robotic_pose, open(save_path, "w"), indent=4)

    def save_interpolated_trans_list(self, save_path):
        json.dump(self.interpolated_trans_list, open(save_path, "w"), indent=4)

#
# # The Interpolator used to create the area around a certain sampling point
# class RoboticSurfaceInterpolator:
#     def __init__(self):
#         self.pose_list = []
#         self.points_list = []
#
#         self.referenced_xy = numpy.array([[0, 0]])
#         self.referenced_z = numpy.array([[0]])
#
#     def get_interpolated_value(self, x, y):
#         # grid_x, grid_y = np.mgrid[-2:2:100j, -2:2:100j]
#         interpolated_z = griddata(self.referenced_xy, self.referenced_z, (x, y), method='cubic')
#         return interpolated_z


# def rotation_matrix(normal):
#     a = numpy.asarray(normal)
#     [x, y, z] = (a / numpy.linalg.norm(a)).tolist()
#     projected_xy = x*x + y*y
#     if projected_xy != 0:
#         rot_x = math.acos(z)
#         rot_z = math.asin(x / math.sqrt(projected_xy))
#         if y < 0:
#             rot_z = math.pi - rot_z
#         [n_z, n_x, n_y] = Rotation.from_euler("xzy", [-rot_x, -rot_z, 0]).as_euler("zxy")
#         rotation = Rotation.from_euler("zxy", [0, n_x, n_y]).as_dcm()
#     else:
#         if z >= 0:
#             rotation = numpy.identity(3)
#         else:
#             rotation = Rotation.from_euler("zxy", [0, math.pi, 0]).as_dcm()
#     return rotation

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




# def interpolate_surface_pose_list(pose_list):
#     points_all = []
#     normals_all = []
#     colors_all = []
#
#     for i, pose in enumerate(pose_list):
#         points
#
#
#
#
# # preparing data
# original_pose = numpy.array([[1, 0, 0, 0],
#                              [0, 1, 0, 0],
#                              [0, 0, 1, 2],
#                              [0, 0, 0, 1]])
# pose_list = []
# pcd_all = PointCloud()
#
# tile_info_dict = {}
#
# rotations_eul = numpy.random.randint(0, 200, size=(400, 3)) / 100.0
# # print(rotations_eul)
# #
# for i, eu in enumerate(rotations_eul):
#     pose = numpy.dot(transforms3d.affines.compose([0, 0, 0], Rotation.from_euler("xyz", eu).as_dcm(), [1, 1, 1]),
#                      numpy.asarray(original_pose))
#
#     pose_list.append(pose)
#     pcd = PointCloud()
#     pcd.points = Vector3dVector(numpy.asarray([[0, 0, 0]]))
#     pcd.colors = Vector3dVector(numpy.asarray([[0, 1, 0]]))
#     pcd.normals = Vector3dVector(numpy.asarray([[0, 0, 1]]))
#     pcd.transform(pose)
#     pcd_all += pcd
#
#     tile_info_dict[i] = TileInfo(tile_index=i,
#                                  tile_info_path="",
#                                  image_path="",
#                                  zoom_level=0,
#                                  position=[0, 0, 0], rotation=[0, 0, 1, 0])
#     tile_info_dict[i].pose_matrix = pose
#     tile_info_dict[i].position = pcd.points[0]
#
# # draw_geometries([pcd_all])
#
# # try to build the surface
# _, tile_tree, tile_index_list = tile_info_dict_generate_kd_tree(tile_info_dict)
#
#
# def interpolating_for_tile(id, radius, tile_info_dict, tile_tree, tile_index_list):
#     tile_info = tile_info_dict[id]
#     [_, idx, _] = tile_tree.search_radius_vector_3d(tile_info.position, radius=radius)
#     normalized_positions = []
#     for index in idx:
#         key = tile_index_list[index]
#         # in_range_key_list.append(key)
#         posistion = numpy.dot(numpy.linalg.inv(tile_info.pose_matrix),
#                               numpy.dot(tile_info_dict[key].pose_matrix,
#                                         numpy.array([0, 0, 0, 1]).T)).T[0:3]
#         normalized_positions.append(posistion)
#
#     # print(normalized_positions)
#
#     normalized_positions = numpy.asarray(normalized_positions)
#     # print(normalized_positions)
#     points = normalized_positions[:,0:2]
#     values = normalized_positions[:,2:3]
#     # print(points)
#     # print(values)
#     grid_x, grid_y = np.mgrid[-2:2:100j, -2:2:100j]
#     grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
#
#     print(numpy.asarray(grid_x).reshape((1,-1)))
#     pos_list = numpy.asarray(
#         [numpy.asarray(grid_x).reshape((1,-1))[0],
#         numpy.asarray(grid_y).reshape((1,-1))[0],
#         numpy.asarray(grid_z).reshape((1,-1))[0]]).T
#     # print(type(pos_list[0][2]))
#     # if math.isnan(pos_list[0][2]):
#     #     print("is nan")
#     points = pos_list[numpy.logical_not(numpy.isnan(pos_list[:, 2]))]
#     print(len(pos_list))
#     print(points)
#
#     pcd = PointCloud()
#     pcd.points = Vector3dVector(points)
#     pcd.colors = Vector3dVector(numpy.repeat([[1, 0, 0]], len(normalized_positions), axis=0))
#
#     geometry.estimate_normals(pcd, search_param=geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
#
#     normal_list = []
#     for normal in pcd.normals:
#         if normal[2] < 0:
#             normal_list.append(normal * -1)
#         else:
#             normal_list.append(normal)
#     pcd.normals = Vector3dVector(normal_list)
#
#     return pcd
#
#
#
# # interpolating_for_tile(0, 1, tile_info_dict, tile_tree, tile_index_list)
#
# id = 5
# radius = 2
# pcd_in_range = interpolating_for_tile(id, radius, tile_info_dict, tile_tree, tile_index_list)
# pcd_in_range.transform(numpy.array([[1, 0, 0, 0],
#                                     [0, 1, 0, 0],
#                                     [0, 0, 1, -0.01],
#                                     [0, 0, 0, 1]]))
# pcd_in_range.transform(tile_info_dict[id].pose_matrix)
#
# pcd_center = PointCloud()
# print(tile_info_dict[id].position)
# pcd_center.points = Vector3dVector([tile_info_dict[id].position])
# pcd_center.colors = Vector3dVector([[0, 0, 1]])
#
# # all = pcd_all+pcd_in_range+pcd_center
# # geometry.estimate_normals(all, search_param=geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
#
# # draw_geometries([all])
#
# draw_geometries([pcd_all, pcd_in_range, pcd_center])