import numpy
import transforms3d
import json
import open3d
import file_managing
import os.path
from scipy.spatial.transform import Rotation

trans_rob_to_camera = transforms3d.affines.compose(T=[0, 0, 0],
                                                   R=[[-1, 0, 0],
                                                      [0, -1, 0],
                                                      [0, 0, -1]],
                                                   Z=[1, 1, 1])
plane_shifting_rotation_trans = numpy.array([[0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [1, 0, 0, 0],
                                             [0, 0, 0, 1]])
trans_rob_to_camera = numpy.dot(trans_rob_to_camera, plane_shifting_rotation_trans)
trans_camera_to_rob = numpy.linalg.inv(trans_rob_to_camera)


def rob_pose_to_trans(rob_pose):
    return numpy.dot(numpy.asarray(rob_pose).reshape((4, 4)).T, trans_rob_to_camera)


def trans_to_rob_pose(trans):
    return numpy.dot(numpy.asarray(trans), trans_camera_to_rob).T.reshape((-1)).tolist()


def read_robotic_pose_list_as_trans_list(path, exclude_beginning_n=0):
    robotic_pose_list = json.load(open(path, "r"))
    trans_list = []
    for i, pose in enumerate(robotic_pose_list):
        if i >= exclude_beginning_n:
            trans_list.append(rob_pose_to_trans(pose))
    return trans_list


def read_robotic_pose_list(path, exclude_beginning_n=0):
    robotic_pose_list = json.load(open(path, "r"))
    robotic_pose_list = robotic_pose_list[exclude_beginning_n:]
    return robotic_pose_list


def read_trans_list(path, exclude_beginning_n=0):
    trans_data_list = json.load(open(path, "r"))
    trans_list = []
    for i, trans_data in enumerate(trans_data_list):
        if i >= exclude_beginning_n:
            trans_list.append(numpy.asarray(trans_data))
    return trans_list


def make_robotic_pose_list(robotic_pose_dir_path):
    # Read all pose files from the directory.
    if os.path.isdir(robotic_pose_dir_path):
        pose_file_list = file_managing.get_file_list(robotic_pose_dir_path, extension=".json")
    else:
        print(robotic_pose_dir_path + " is not a directory")
        return None
    robotic_pose_list = []
    trans_list = []
    for pose_file_path in pose_file_list:
        robotic_pose_data = json.load(open(pose_file_path, "r"))
        trans_data = rob_pose_to_trans(robotic_pose_data)

        robotic_pose_list.append(robotic_pose_data)
        trans_list.append(trans_data)
    return robotic_pose_list, trans_list


def save_robotic_pose_list(path, robotic_pose_list):
    json.dump(robotic_pose_list, open(path, "w"), indent=4)


def save_trans_list(path, trans_list):
    data_to_save = []
    for trans in trans_list:
        data_to_save.append(trans.tolist())
    json.dump(data_to_save, open(path, "w"), indent=4)


def save_trans_as_robotic_pose(path, trans_list):
    robotic_pose_list = []
    for trans in trans_list:
        robotic_pose_list.append(trans_to_rob_pose(trans))
    json.dump(robotic_pose_list, open(path, "w"), indent=4)


def save_robotic_full_pose_list(path, robotic_full_pose_list):
    json.dump(robotic_full_pose_list, open(path, "w"), indent=4)


# can also be used to make a sampled list.
def adjust_order(source_list, index_list):
    target_list = []
    for index in index_list:
        target_list.append(source_list[index])
    return target_list


def trans_list_to_points(trans_list):
    point_list = []
    for trans in trans_list:
        point = numpy.dot(trans, numpy.asarray([0, 0, 0, 1]).T).T[0:3].tolist()
        point_list.append(point)
    return point_list


def trans_list_to_points_normals(trans_list, original_normal=[1.0, 0, 0]):
    point_list = []
    normal_list = []
    for trans in trans_list:
        point = numpy.dot(trans, numpy.asarray([0, 0, 0, 1]).T).T[0:3].tolist()
        normal = numpy.dot(trans, numpy.asarray([original_normal[0],
                                                 original_normal[1],
                                                 original_normal[2],
                                                 0]).T).T[0:3].tolist()
        point_list.append(point)
        normal_list.append(normal)
    return point_list, normal_list


def robotic_pose_list_to_points(robotic_pose_list):
    point_list = []
    for pose in robotic_pose_list:
        trans = rob_pose_to_trans(pose)
        point = numpy.dot(trans, numpy.asarray([0, 0, 0, 1]).T).T[0:3].tolist()
        point_list.append(point)
    return point_list


def trans_list_to_pos_ori_list(trans_list):
    robotic_full_pose_list = []
    for trans in trans_list:
        trans = numpy.dot(trans_camera_to_rob, trans)
        [T, R, Z, S] = transforms3d.affines.decompose(trans)
        quaternion = Rotation.from_dcm(R).as_quat()
        robotic_full_pose_list.append([T.tolist(), quaternion.tolist()])
    return robotic_full_pose_list


def non_outlier_index_list_static(point_list, nb_neighbours=20, std_ratio=2.0):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(point_list)

    cl, ind = open3d.geometry.statistical_outlier_removal(pcd,
                                                          nb_neighbors=nb_neighbours,
                                                          std_ratio=std_ratio)
    display_inlier_outlier(pcd, ind)


def non_outlier_index_list_radius(point_list, nb_points=20, radius=0.01):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(point_list)

    cl, ind = open3d.geometry.radius_outlier_removal(pcd,
                                                     nb_points=nb_points,
                                                     radius=radius)
    display_inlier_outlier(pcd, ind)


def display_inlier_outlier(cloud, ind):
    inlier_cloud = open3d.geometry.select_down_sample(cloud, ind)
    outlier_cloud = open3d.geometry.select_down_sample(cloud, ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

