import numpy
import cv2
from scipy.spatial.transform import Rotation
import transforms3d
import sys
sys.path.append("../Utility")
sys.path.append("../CoreFunctionModules")
from tile_info_processing import *
import multiprocessing
from joblib import Parallel, delayed


class LocalTransEstimation:
    def __init__(self, s, t, success, conf, trans):
        self.s = s
        self.t = t
        self.success = success
        self.conf = conf
        self.trans = trans


class TransDataG2o:
    # def __init__(self):
    #     self.trans_dict = {}
    def __init__(self, tile_info_dict, config):
        self.trans_dict = {}
        self.tile_info_dict = tile_info_dict
        self.config = config

    def update_trans(self, s_id, t_id, success, conf, trans):
        self.trans_dict[(s_id, t_id)] = LocalTransEstimation(s_id, t_id, success, conf, trans)

    def get_trans(self, s_id, t_id):
        try:
            local_trans_estimation = self.trans_dict[(s_id, t_id)]
        except:
            try:
                s_info = self.tile_info_dict[s_id]
                t_info = self.tile_info_dict[t_id]
                success, conf, trans = trans_estimation(s_info, t_info, self.config)
                self.update_trans(s_id, t_id, success, conf, trans)
                local_trans_estimation = self.trans_dict[(s_id, t_id)]
            except:
                local_trans_estimation = LocalTransEstimation(s_id, t_id, False, 0, numpy.identity(4))
        return local_trans_estimation

    def get_trans_extend(self, s_id, t_id):
        l_trans_e = self.get_trans(s_id, t_id)
        return l_trans_e.success, l_trans_e.conf, l_trans_e.trans

    def update_local_trans_data_multiprocessing(self):
        # make estimation list:
        estimation_list = []
        for s_id in self.tile_info_dict:
            tile_info = self.tile_info_dict[s_id]
            for t_id in tile_info.odometry_list:
                estimation_list.append([s_id, t_id])
            for t_id in tile_info.potential_loop_closure:
                estimation_list.append([s_id, t_id])

        max_thread = min(multiprocessing.cpu_count(), max(len(estimation_list), 1))

        # There might be an efficiency issue. the function self.get_trans(s, t)
        # It might be better if there is a independent function.
        # Currently, the multiprocessing does have multiple threads, but there is only one core working.
        # it is possible that some part of the class instance cannot be shared between cores.
        estimation_results = Parallel(n_jobs=max_thread)(
            delayed(trans_estimation_pure)(s_id, t_id,
                                           self.tile_info_dict[s_id].image_path,
                                           self.tile_info_dict[t_id].image_path,
                                           self.tile_info_dict[s_id].width_by_pixel,
                                           self.tile_info_dict[s_id].height_by_pixel,
                                           self.tile_info_dict[t_id].width_by_pixel,
                                           self.tile_info_dict[t_id].height_by_pixel,
                                           self.tile_info_dict[s_id].width_by_mm,
                                           self.tile_info_dict[s_id].height_by_mm,
                                           self.tile_info_dict[t_id].width_by_mm,
                                           self.tile_info_dict[t_id].height_by_mm,
                                           self.config["crop_width_by_pixel"],
                                           self.config["crop_height_by_pixel"],
                                           self.tile_info_dict[s_id].init_transform_matrix,
                                           self.tile_info_dict[t_id].init_transform_matrix,
                                           self.config["n_features"],
                                           self.config["num_matches_thresh1"],
                                           self.config["conf_threshold"],
                                           self.config["scaling_tolerance"],
                                           self.config["rotation_tolerance"])
            for [s_id, t_id] in estimation_list)

        for result in estimation_results:
            self.update_trans(s_id=result[0],
                              t_id=result[1],
                              success=result[2],
                              conf=result[3],
                              trans=result[4])

    def update_tile_info_dict_confirmed_loop_closure(self):
        for tile_info_key in self.tile_info_dict:
            tile_info = self.tile_info_dict[tile_info_key]
            for potential_loop_closure_t in tile_info.potential_loop_closure:
                if self.get_trans(tile_info.tile_index, potential_loop_closure_t).success:
                    if potential_loop_closure_t not in tile_info.confirmed_loop_closure:
                        tile_info.confirmed_loop_closure.append(potential_loop_closure_t)
        return self.tile_info_dict

    def save(self, path=None):
        if path is None:
            path = join(self.config["path_data"], self.config["local_trans_dict_g2o"])
        data_to_save = {}
        for (s, t) in self.trans_dict:
            try:
                data_to_save[s][t] = {"s": self.trans_dict[(s, t)].s,
                                      "t": self.trans_dict[(s, t)].t,
                                      "success": self.trans_dict[(s, t)].success,
                                      "conf": self.trans_dict[(s, t)].conf,
                                      "trans": self.trans_dict[(s, t)].trans.tolist()}
            except:
                data_to_save[s] = {}
                data_to_save[s][t] = {"s": self.trans_dict[(s, t)].s,
                                      "t": self.trans_dict[(s, t)].t,
                                      "success": self.trans_dict[(s, t)].success,
                                      "conf": self.trans_dict[(s, t)].conf,
                                      "trans": self.trans_dict[(s, t)].trans.tolist()}

        json.dump(data_to_save, open(path, "w"), indent=4)

    def read(self, path=None):
        if path is None:
            path = join(self.config["path_data"], self.config["local_trans_dict_g2o"])
        readed_data = json.load(open(path, "r"))
        for s in readed_data:
            for t in readed_data[s]:
                self.update_trans(s_id=int(readed_data[s][t]["s"]),
                                  t_id=int(readed_data[s][t]["t"]),
                                  success=bool(readed_data[s][t]["success"]),
                                  conf=float(readed_data[s][t]["conf"]),
                                  trans=numpy.asarray(readed_data[s][t]["trans"]))


def trans_estimation(s_info: TileInfo, t_info: TileInfo, config):
    s_img = cv2.imread(s_info.image_path)
    t_img = cv2.imread(t_info.image_path)

    matching_conf, trans_cv_2d = transformation_cv(s_img, t_img,
                                                   crop_w=config["crop_width_by_pixel"],
                                                   crop_h=config["crop_height_by_pixel"],
                                                   nfeatures=config["n_features"],
                                                   num_matches_thresh1=config["num_matches_thresh1"],
                                                   match_conf=config["conf_threshold"])

    if matching_conf == 0:
        return False, 0, numpy.identity(4)

    trans_planar_3d = transform_convert_from_2d_to_3d(trans_cv_2d=trans_cv_2d,
                                                      width_by_pixel_s=s_info.width_by_pixel,
                                                      height_by_pixel_s=s_info.height_by_pixel,
                                                      width_by_mm_s=s_info.width_by_mm,
                                                      height_by_mm_s=s_info.height_by_mm,
                                                      width_by_pixel_t=t_info.width_by_pixel,
                                                      height_by_pixel_t=t_info.height_by_pixel,
                                                      width_by_mm_t=t_info.width_by_mm,
                                                      height_by_mm_t=t_info.height_by_mm)

    if not trans_local_check(trans_planar_3d, s_info.init_transform_matrix, t_info.init_transform_matrix,
                             scaling_tolerance=config["scaling_tolerance"],
                             rotation_tolerance=config["rotation_tolerance"]):
        return False, 0, numpy.identity(4)

    trans_3d = transform_planar_add_normal_direction(trans_planar_3d,
                                                     s_info.init_transform_matrix, t_info.init_transform_matrix)
    return True, matching_conf, trans_3d


def trans_estimation_pure(s_id, t_id,
                          s_img_path,
                          t_img_path,
                          width_by_pixel_s,
                          height_by_pixel_s,
                          width_by_pixel_t,
                          height_by_pixel_t,
                          width_by_mm_s,
                          height_by_mm_s,
                          width_by_mm_t,
                          height_by_mm_t,
                          crop_w,
                          crop_h,
                          s_init_trans,
                          t_init_trans,
                          n_features,
                          num_matches_thresh1,
                          match_conf,
                          scaling_tolerance,
                          rotation_tolerance):

    s_img = cv2.imread(s_img_path)
    t_img = cv2.imread(t_img_path)

    matching_conf, trans_cv_2d = transformation_cv(s_img, t_img,
                                                   crop_w=crop_w,
                                                   crop_h=crop_h,
                                                   nfeatures=n_features,
                                                   num_matches_thresh1=num_matches_thresh1,
                                                   match_conf=match_conf)

    if matching_conf == 0:
        return s_id, t_id, False, 0, numpy.identity(4)

    trans_planar_3d = transform_convert_from_2d_to_3d(trans_cv_2d=trans_cv_2d,
                                                      width_by_pixel_s=width_by_pixel_s,
                                                      height_by_pixel_s=height_by_pixel_s,
                                                      width_by_mm_s=width_by_mm_s,
                                                      height_by_mm_s=height_by_mm_s,
                                                      width_by_pixel_t=width_by_pixel_t,
                                                      height_by_pixel_t=height_by_pixel_t,
                                                      width_by_mm_t=width_by_mm_t,
                                                      height_by_mm_t=height_by_mm_t)

    if not trans_local_check(trans_planar_3d, s_init_trans, t_init_trans,
                             scaling_tolerance=scaling_tolerance,
                             rotation_tolerance=rotation_tolerance):
        print("Tile %05d and %05d : Local trans check fails " % (s_id, t_id))
        return s_id, t_id, False, matching_conf, trans_planar_3d
    else:
        print("Tile %05d and %05d : Trans check passed" % (s_id, t_id))
        trans_3d = transform_planar_add_normal_direction(trans_planar_3d, s_init_trans, t_init_trans)
        return s_id, t_id, True, matching_conf, trans_3d


# Can only take color images
def transformation_cv(s_img, t_img, crop_w=0, crop_h=0,
                      nfeatures=400, num_matches_thresh1=6, match_conf=0.3):
    (s_h, s_w, s_c) = s_img.shape
    (t_h, t_w, t_c) = t_img.shape
    s_img_crop = s_img[crop_h:s_h - crop_h, crop_w:s_w - crop_w]
    t_img_crop = t_img[crop_h:t_h - crop_h, crop_w:t_w - crop_w]

    orb_finder = cv2.ORB_create(scaleFactor=1.2, nlevels=4, edgeThreshold=31,
                                firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
                                nfeatures=nfeatures, patchSize=31)
    matcher = cv2.detail_AffineBestOf2NearestMatcher(full_affine=False, try_use_gpu=True,
                                                     match_conf=match_conf,
                                                     num_matches_thresh1=num_matches_thresh1)
    source_feature = cv2.detail.computeImageFeatures2(featuresFinder=orb_finder, image=s_img_crop)
    target_feature = cv2.detail.computeImageFeatures2(featuresFinder=orb_finder, image=t_img_crop)
    matching_result = matcher.apply(source_feature, target_feature)
    matcher.collectGarbage()
    # print(type(matching_result))

    match_conf = matching_result.confidence

    if match_conf != 0.0:
        trans_cv = numpy.asarray(matching_result.H)

        trans_deoffset = numpy.asarray([[1.0, 0, crop_w],
                                        [0, 1.0, crop_h],
                                        [0, 0, 1]])
        trans_offset = numpy.asarray([[1.0, 0, -crop_w],
                                      [0, 1.0, -crop_h],
                                      [0, 0, 1]])
        trans_cv = numpy.dot(trans_deoffset, numpy.dot(trans_cv, trans_offset))
    else:
        trans_cv = numpy.identity(4)
    return match_conf, trans_cv


def transform_convert_from_2d_to_3d(trans_cv_2d,
                                    width_by_pixel_s=320, height_by_pixel_s=240,
                                    width_by_mm_s=3.2, height_by_mm_s=1.6,
                                    width_by_pixel_t=640, height_by_pixel_t=480,
                                    width_by_mm_t=6.4, height_by_mm_t=4.8):

    scaling_compensate_x = (width_by_mm_t / width_by_pixel_t) / (width_by_mm_s / width_by_pixel_s)
    scaling_compensate_y = (height_by_mm_t / height_by_pixel_t) / (height_by_mm_s / height_by_pixel_s)

    trans_compensate_x = width_by_mm_t / width_by_pixel_t
    trans_compensate_y = height_by_mm_t / height_by_pixel_t

    compensate_factors = numpy.asarray([[scaling_compensate_x, scaling_compensate_y, 0, trans_compensate_x],
                                        [scaling_compensate_x, scaling_compensate_y, 0, trans_compensate_y],
                                        [                   0,                    0, 1,                  0],
                                        [                   0,                    0, 0,                  1]])

    trans_cv_four = numpy.asarray([[ trans_cv_2d[0][0], -trans_cv_2d[0][1], 0,  trans_cv_2d[0][2]],
                                   [-trans_cv_2d[1][0],  trans_cv_2d[1][1], 0,  -trans_cv_2d[1][2]],
                                   [              0,               0, 1,               0],
                                   [              0,               0, 0,               1]])
    trans_cv_3d = trans_cv_four * compensate_factors

    # (ct, cr, cz, cs) = transforms3d.affines.decompose44(trans_cv_four)

    trans_deoffset = [[1, 0, 0, -width_by_mm_t/2],
                     [0, 1, 0, height_by_mm_t/2],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
    trans_offset = [[1, 0, 0, width_by_mm_s/2],
                    [0, 1, 0, -height_by_mm_s/2],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]

    trans_planar_3d = numpy.dot(numpy.dot(trans_deoffset, trans_cv_3d), trans_offset)
    return trans_planar_3d


def transform_planar_add_normal_direction(trans_planar, trans_s, trans_t):
    trans_rotation = numpy.dot(numpy.linalg.inv(trans_t), trans_s)

    (pt, pr, pz, ps) = transforms3d.affines.decompose44(trans_planar)
    (rt, rr, rz, rs) = transforms3d.affines.decompose44(trans_rotation)

    rr_euler = Rotation.from_dcm(rr).as_euler("xyz")  # should have x and y value.
    # trans_one = transforms3d.affines.compose(pt / 2, pr, pz, ps)
    trans_one = transforms3d.affines.compose(pt / 2, pr, [1, 1, 1])
    trans_two = transforms3d.affines.compose([0, 0, 0],
                                             Rotation.from_euler("xyz", [rr_euler[0], rr_euler[1], 0]).as_dcm(),
                                             [1, 1, 1])
    trans_three = transforms3d.affines.compose(pt / 2, numpy.identity(3), [1, 1, 1])
    trans_with_normal_direction = numpy.dot(numpy.dot(trans_three, trans_two), trans_one)

    return trans_with_normal_direction


# def trans_local_check_tile(trans_local, s_info, t_info, scaling_tolerance=0.05, rotation_tolerance=0.2):
#     trans_diff = numpy.dot(
#         numpy.linalg.inv(numpy.dot(s_info.init_transform_matrix, numpy.linalg.inv(t_info.init_transform_matrix))),
#         trans_local)
#
#     (pt, pr, pz, ps) = transforms3d.affines.decompose44(trans_diff)
#     rotation_euler_z = Rotation.from_dcm(pr).as_euler("xyz")[2]
#
#     if abs(pz[0] - 1) > scaling_tolerance \
#             or abs(pz[1] - 1) > scaling_tolerance \
#             or abs(pz[2] - 1) > scaling_tolerance:
#         print("Tile %05d and %05d : Scaling check fails for (%4f, %4f, %4f)"
#               % (s_info.tile_index, t_info.tile_index, pz[0], pz[1], pz[2]))
#         return False
#     if abs(rotation_euler_z) > rotation_tolerance:
#         print("Tile %05d and %05d : Rotation check fails for (%4f)"
#               % (s_info.tile_index, t_info.tile_index, rotation_euler_z))
#         return False
#     return True


def trans_local_check(trans_local, s_init_trans, t_init_trans, scaling_tolerance=0.05, rotation_tolerance=0.2):
    trans_diff = numpy.dot(
        numpy.linalg.inv(numpy.dot(s_init_trans, numpy.linalg.inv(t_init_trans))), trans_local)

    (pt, pr, pz, ps) = transforms3d.affines.decompose44(trans_diff)
    rotation_euler_z = Rotation.from_dcm(pr).as_euler("xyz")[2]

    if abs(pz[0] - 1) > scaling_tolerance \
            or abs(pz[1] - 1) > scaling_tolerance \
            or abs(pz[2] - 1) > scaling_tolerance:
        return False
    if abs(rotation_euler_z) > rotation_tolerance:
        return False
    return True


def trans_info_matching_g2o(match_conf, weight=1):
    # The order should be: x, y, z, rotation_x, rotation_y, rotation_z
    info_matrix = weight * match_conf * numpy.asarray([[3, 0, 0, 0, 0, 0],
                                                       [0, 3, 0, 0, 0, 0],
                                                       [0, 0, 2, 0, 0, 0],
                                                       [0, 0, 0, 2, 0, 0],
                                                       [0, 0, 0, 0, 2, 0],
                                                       [0, 0, 0, 0, 0, 3]])
    return info_matrix


def trans_info_sensor_g2o(weight=1):
    # The order should be: x, y, z, rotation_x, rotation_y, rotation_z
    info_matrix = weight * numpy.asarray([[0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 5, 0, 0],
                                          [0, 0, 0, 0, 5, 0],
                                          [0, 0, 0, 0, 0, 5]])
    return info_matrix
