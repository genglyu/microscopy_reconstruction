import cv2
from open3d import *
import numpy
import scipy.spatial.transform
import transforms3d

import sys
sys.path.append("../Utility")


def load_image_as_numpy_array(image_bgr, cv_scale_factor=-1, convert_to_indensity=False):
    loaded_image_numpy_array = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # OpenCV use BGR color.

    if cv_scale_factor != -1:
        loaded_image_numpy_array = cv2.resize(loaded_image_numpy_array, None, fx=cv_scale_factor, fy=cv_scale_factor)

    if convert_to_indensity:
        loaded_image_numpy_array = cv2.cvtColor(loaded_image_numpy_array, cv2.COLOR_RGB2GRAY)
        loaded_image_numpy_array = cv2.cvtColor(loaded_image_numpy_array, cv2.COLOR_GRAY2RGB)

    return loaded_image_numpy_array


def load_image_as_planar_point_cloud_open3d(image_bgr, width_by_mm, height_by_mm,
                                            cv_scale_factor=-1, convert_to_indensity=False):
    img_numpy_array = load_image_as_numpy_array(image_bgr, cv_scale_factor=cv_scale_factor,
                                                convert_to_indensity=convert_to_indensity)
    (height_by_pixel, width_by_pixel, value_length) = img_numpy_array.shape

    pixel_amount = height_by_pixel * width_by_pixel
    points_position = numpy.mgrid[(height_by_pixel / 2):(-height_by_pixel + height_by_pixel / 2):-1,
                      (-width_by_pixel / 2):(width_by_pixel - width_by_pixel / 2):1].reshape(2, -1).T
    # ================================================================================================
    points_position = numpy.c_[
        points_position[:, 1] * width_by_mm / width_by_pixel,
        points_position[:, 0] * height_by_mm / height_by_pixel,
        numpy.zeros(pixel_amount)
    ]

    points_normal = numpy.c_[
        numpy.zeros(pixel_amount), numpy.zeros(pixel_amount), numpy.ones(pixel_amount)]
    points_color = img_numpy_array.reshape(-1, 3)
    points_color = points_color / 255.0

    colored_point_cloud = PointCloud()
    colored_point_cloud.points = Vector3dVector(points_position)
    colored_point_cloud.colors = Vector3dVector(points_color)
    colored_point_cloud.normals = Vector3dVector(points_normal)
    colored_point_cloud.normalize_normals()
    # if voxel_size != -1:
    #     colored_point_cloud = voxel_down_sample(colored_point_cloud, voxel_size)
    # ================================================================================================
    return colored_point_cloud
