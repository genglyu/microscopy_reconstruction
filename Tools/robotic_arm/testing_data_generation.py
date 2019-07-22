import numpy
from open3d import *
from scipy.spatial.transform import *
import transforms3d
import json
from robotic_surface_interpolation import *


# preparing data
original_pose = numpy.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0.02],
                             [0, 0, 0, 1]])

rotations_eul = numpy.random.randint(-60, 60, size=(800, 2)) / 100.0
# rotations_eul = numpy.random.randint(0, 200, size=(400, 3)) / 100.0
# print(rotations_eul)
#

pose_list = []

for i, eu in enumerate(rotations_eul):
    pose = numpy.dot(transforms3d.affines.compose([0, 0, 0],
                                                  Rotation.from_euler("zxy", [0, eu[0], eu[1]]).as_dcm(),
                                                  [1, 1, 1]),
                     numpy.asarray(original_pose))
    # pose = numpy.dot(transforms3d.affines.compose([0, 0, 0],
    #                                               Rotation.from_euler("xyz", eu).as_dcm(),
    #                                               [1, 1, 1]),
    #                  numpy.asarray(original_pose))

    robotic_pos = trans_to_rob_pose(pose)
    pose_list.append(robotic_pos)

# data_to_save = {"pose_list": pose_list}
data_to_save = pose_list
json.dump(data_to_save, open("testing_pose.testingjson", "w"), indent=4)

