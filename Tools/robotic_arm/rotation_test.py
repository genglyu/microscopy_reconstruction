import transforms3d
import numpy
from scipy.spatial.transform import Rotation
import json
from open3d import *
import math

trans_rob_to_camera = transforms3d.affines.compose([0, 0, 0], Rotation.from_euler("xyz", [0, 0, 0]).as_dcm(), [1, 1, 1])
trans_camera_to_rob = numpy.linalg.inv(trans_rob_to_camera)


def rob_pose_to_trans(rob_pose):
    pose = numpy.dot(numpy.asarray(rob_pose).reshape((4, 4)).T, trans_rob_to_camera)
    # pose = pose * numpy.array()
    return pose


def trans_to_rob_pose(trans):
    return numpy.dot(numpy.asarray(trans), trans_camera_to_rob).T.reshape((-1)).tolist()


def rotation_matrix(normal):
    a = numpy.asarray(normal)
    [x, y, z] = (a / numpy.linalg.norm(a)).tolist()
    projected_xy = x*x + y*y
    if projected_xy != 0:
        rot_x = math.acos(z)
        rot_z = math.asin(x / math.sqrt(projected_xy))
        if y < 0:
            rot_z = math.pi - rot_z
        [n_z, n_x, n_y] = Rotation.from_euler("xzy", [-rot_x, -rot_z, 0]).as_euler("zxy")
        rotation = Rotation.from_euler("zxy", [0, n_x, n_y]).as_dcm()
    else:
        if z >= 0:
            rotation = numpy.identity(3)
        else:
            rotation = Rotation.from_euler("zxy", [0, math.pi, 0]).as_dcm()
    return rotation


def make_tile_frame(trans_matrix, width, height, color=[0.5, 0.5, 0.5]):
    tile_frame = LineSet()
    lb_rb_rt_lt = [[-width / 2, -height / 2, 0],
                   [ width / 2, -height / 2, 0],
                   [ width / 2,  height / 2, 0],
                   [-width / 2,  height / 2, 0]
                   ]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    colors = [[1, 0, 0], [0, 0, 1], color, color]
    tile_frame.points = Vector3dVector(lb_rb_rt_lt)
    tile_frame.lines = Vector2iVector(lines)
    tile_frame.colors = Vector3dVector(colors)
    tile_frame.transform(trans_matrix)
    return tile_frame


robotic_pose_list = json.load(open("testing_pose_2.testingjson", "r"))


for robotic_pose in robotic_pose_list:
    pose = rob_pose_to_trans(robotic_pose)
    pos = numpy.dot(pose, numpy.array([0, 0, 0, 1]).T).T[0:3].tolist()
    normal = numpy.dot(pose, numpy.array([0, 0, 1, 0]).T).T[0:3]

    rot = rotation_matrix(normal)

    print(normal)
    print(numpy.dot(rot, numpy.asarray([0, 0, 1])))

    trans = transforms3d.affines.compose(T=pos, R=rot, Z=[1, 1, 1])

    source_tile = make_tile_frame(pose, width=0.04, height=0.03, color=[0.5, 0.5, 0.5])
    converted = make_tile_frame(trans, width=0.04, height=0.03, color=[0, 1, 0])

    coor = geometry.create_mesh_coordinate_frame(size=0.05, origin=pos)

    draw_geometries([source_tile, converted, coor])




