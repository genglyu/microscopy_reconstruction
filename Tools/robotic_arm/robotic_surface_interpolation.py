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
from tile_info_processing import *
import visualization


trans_rob_to_camera = transforms3d.affines.compose([0, 0, 0], Rotation.from_euler("xyz", [0, 0, 0]).as_dcm(), [1, 1, 1])
trans_camera_to_rob = numpy.linalg.inv(trans_rob_to_camera)


def rob_pose_to_trans(rob_pose):
    return numpy.dot(numpy.asarray(rob_pose).reshape((4, 4)).T, trans_rob_to_camera)


def trans_to_rob_pose(trans):
    return numpy.dot(numpy.asarray(trans), trans_camera_to_rob).T.reshape((-1)).tolist()


class RoboticSurfaceConstructor:
    def __init__(self):
        self.pose_list = []
        self.points_list = []
        self.sampling_pcd = PointCloud()
        self.sampling_kd_tree = None

        self.interpolated_pose_list = []
        self.interpolate_robotic_pose = []
        self.integrated_surface_pcd = PointCloud()
    # Find the closest actually sampled point and then add rotation to the transformation.

    def read_robotic_pose_list(self, robotic_pose_list_path):
        # robotic_pose_list = json.load(open(robotic_pose_list_path, "r"))["pose_list"]
        robotic_pose_list = json.load(open(robotic_pose_list_path, "r"))
        for i, robotic_pose in enumerate(robotic_pose_list):
            pose = rob_pose_to_trans(robotic_pose)
            self.pose_list.append(pose)
            self.points_list.append(numpy.dot(pose, numpy.array([0, 0, 0, 1]).T).T[0:3].tolist())

        self.points_list = numpy.asarray(self.points_list)

        self.sampling_pcd.points = Vector3dVector(numpy.asarray(self.points_list))
        self.sampling_pcd.colors = Vector3dVector(numpy.repeat(numpy.array([[0, 0, 0]]), len(self.points_list), axis=0))
        self.sampling_kd_tree = KDTreeFlann(self.sampling_pcd)

        draw_geometries([self.sampling_pcd])

    def interpolate_sub(self, index=0, radius=0.03, interpolate_w=0.004, interpolate_h=0.003, interpolate_amount=100):
        [_, idx, _] = self.sampling_kd_tree.search_radius_vector_3d(self.points_list[index], radius=radius)

        idx = list(idx)
        print("interpolate for %d from %d in range points" % (index, len(idx)))

        sub_points = numpy.c_[self.points_list[idx, :], numpy.ones(len(idx))]

        transformed_sub_points = numpy.dot(numpy.linalg.inv(self.pose_list[index]), sub_points.T).T[:, 0:3]

        xy = transformed_sub_points[:, 0:2]
        z = transformed_sub_points[:, 2]

        grid_x, grid_y = numpy.mgrid[
                         -interpolate_w/2:interpolate_w/2:(interpolate_amount * 1j),
                         -interpolate_h/2:interpolate_h/2:(interpolate_amount * 1j)]

        grid_z = griddata(xy, z, (grid_x, grid_y), method='cubic')

        interpolate_points = numpy.c_[grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)]

        interpolate_points = interpolate_points[numpy.logical_not(numpy.isnan(interpolate_points[:, 2]))]

        # print(grid_x)
        # print(grid_x.shape)
        # print(interpolate_points.shape)
        # print(interpolate_points)

        sub_pcd = PointCloud()
        sub_pcd.points = Vector3dVector(interpolate_points)
        geometry.estimate_normals(sub_pcd,
                                  search_param=geometry.KDTreeSearchParamHybrid(radius=interpolate_w, max_nn=30))

        normal_list = []
        for normal in sub_pcd.normals:
            if normal[2] < 0:
                normal_list.append(normal * -1)
            else:
                normal_list.append(normal)
        sub_pcd.normals = Vector3dVector(normal_list)

        sub_pcd.transform(self.pose_list[index])
        # draw_geometries([sub_pcd, self.sampling_pcd])
        return sub_pcd

    def run_interpolation(self):
        for i, pose in enumerate(self.pose_list):
            sub_pcd = self.interpolate_sub(index=i)
            # sub_pcd.transform(pose)
            self.integrated_surface_pcd = self.integrated_surface_pcd + sub_pcd

        # ============================================================
        # self.integrated_surface_pcd = geometry.voxel_down_sample(self.integrated_surface_pcd, voxel_size=0.001)
        self.integrated_surface_pcd = open3d.geometry.uniform_down_sample(input=self.integrated_surface_pcd,
                                                                          every_k_points=100)
        pcd_down, index_in_pcd = \
            geometry.voxel_down_sample_and_trace(input=self.integrated_surface_pcd,
                                                 voxel_size=0.001,
                                                 min_bound=numpy.asarray([0.0001, 0.0001, 0.0001]).T,
                                                 max_bound=numpy.asarray([0.01, 0.01, 0.01]).T,
                                                 approximate_class=False)
        # self.integrated_surface_pcd = pcd_down
        # print(index_in_pcd)
        index_list = index_in_pcd.reshape(-1)
        index_list = numpy.sort(index_list[index_list >= 0]).tolist()

        points = []
        normals = []
        colors = []
        for index in index_list:
            points.append(self.integrated_surface_pcd.points[index])
            normals.append(self.integrated_surface_pcd.normals[index])
            # colors.append(self.integrated_surface_pcd.colors[index])
        self.integrated_surface_pcd.points = Vector3dVector(points)
        self.integrated_surface_pcd.normals = Vector3dVector(normals)
        # self.integrated_surface_pcd.colors = Vector3dVector(colors)

        # ============================================================
        connection = visualization.make_connection_of_pcd_order(self.integrated_surface_pcd)
        draw_geometries([self.integrated_surface_pcd, connection])
        # ============================================================

        for i, position in enumerate(self.integrated_surface_pcd.points):
            rotation_m = rotation_matrix(numpy.array([0, 0, 1]), self.integrated_surface_pcd.normals[i])
            trans = transforms3d.affines.compose(T=position, R=rotation_m, Z=[1, 1, 1])

            self.interpolated_pose_list.append(trans)
            self.interpolate_robotic_pose.append(trans_to_rob_pose(trans))

    def save_interpolated_robotic_pose(self, save_path):
        json.dump(self.interpolate_robotic_pose, open(save_path, "w"), indent=4)

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

def rotation_matrix(normal_s, normal_t):
    a = normal_s / numpy.linalg.norm(normal_s)
    b = normal_t / numpy.linalg.norm(normal_t)
    v = numpy.cross(a, b)
    vx = numpy.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
    rotation = numpy.identity(3) + vx + numpy.dot(vx, vx) * 1 / (1 + numpy.linalg.norm(a * b))
    # In this case, it assumes that the referenced normal direction is the z direction.
    # It would be better if using x direction cause the euler rotations are applied by order x, y, z.
    # The rotation along normal direction is related to image feature matching (z direction in this case.)
    # However, since it is the last one having effect, it also changes the x y rotation values.

    # euler = Rotation.from_dcm(rotation).as_euler("xyz")
    # r_z = Rotation.from_euler("xyz", [0, 0, -euler[2]]).as_dcm()
    # rotation = numpy.dot(rotation, r_z)
    euler = Rotation.from_dcm(rotation).as_euler("zxy")
    rotation = Rotation.from_euler("zxy", [0, euler[1], euler[2]]).as_dcm()
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