import numpy
import cv2

import sys
sys.path.append("../Utility")
sys.path.append("../CoreFunctionModules")


# Can only take color images
def transformation_cv(s_img, t_img, crop_w=0, crop_h=0,
                      nfeatures=400, num_matches_thresh1=6):
    (s_h, s_w, s_c) = s_img.shape
    (t_h, t_w, t_c) = t_img.shape
    s_img_crop = s_img[crop_h:s_h - crop_h, crop_w:s_w - crop_w]
    t_img_crop = t_img[crop_h:t_h - crop_h, crop_w:t_w - crop_w]

    orb_finder = cv2.ORB_create(scaleFactor=1.2, nlevels=4, edgeThreshold=31,
                                firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
                                nfeatures=nfeatures, patchSize=31)
    matcher = cv2.detail_AffineBestOf2NearestMatcher(full_affine=False, try_use_gpu=True,
                                                     match_conf=0.3,
                                                     num_matches_thresh1=num_matches_thresh1)

    source_feature = cv2.detail.computeImageFeatures2(featuresFinder=orb_finder, image=s_img_crop)
    target_feature = cv2.detail.computeImageFeatures2(featuresFinder=orb_finder, image=t_img_crop)
    matching_result = matcher.apply(source_feature, target_feature)

    match_conf = matching_result.confidence
    trans_cv = matching_result.H

    return match_conf, trans_cv


# def transform_convert_from_2d_to_3d(trans_cv,
#                                     width_by_pixel_s=640, height_by_pixel_s=480,
#                                     width_by_mm_s=6.4, height_by_mm_s=4.8,
#                                     width_by_pixel_t=640, height_by_pixel_t=480,
#                                     width_by_mm_t=6.4, height_by_mm_t=4.8):
#     trans_cv_four = [
#             [trans_cv[0][0], -trans_cv[0][1], 0, trans_cv[0][2] / width_by_pixel * width_by_mm],
#             [-trans_cv[1][0], trans_cv[1][1], 0, -trans_cv[1][2] / height_by_pixel * height_by_mm],
#             [0,              0,               1, 0],
#             [0,              0,               0, 1]
#         ]
#     # (ct, cr, cz, cs) = transforms3d.affines.decompose44(trans_cv_four)
#
#     trans_deoffset = [[1, 0, 0, -width_by_mm/2],
#                      [0, 1, 0, height_by_mm/2],
#                      [0, 0, 1, 0],
#                      [0, 0, 0, 1]]
#     trans_offset = [[1, 0, 0, width_by_mm/2],
#                     [0, 1, 0, -height_by_mm/2],
#                     [0, 0, 1, 0],
#                     [0, 0, 0, 1]]
#
#     trans_planar = numpy.dot(numpy.dot(trans_deoffset, trans_cv_four), trans_offset)
#     return trans_planar


