import cv2
import numpy
from apriltag import apriltag
from os import listdir, makedirs
from os.path import exists, isfile, join, splitext

apriltag_img_directory = "/home/lvgeng/Code/Libs/apriltag-imgs"
tag_family_template = {"tag16h5": "tag16_05_%05d.png",
                       "tag25h9": "tag25_09_%05d.png",
                       "tag36h11": "tag36_11_%05d.png",
                       "tagCircle21h7":"tag21_07_%05d.png",
                       "tagCircle49h12": "tag49_12_%05d.png",
                       "tagCustom48h12": "tag48_12_%05d.png",
                       "tagStandard41h12": "tag41_12_%05d.png",
                       "tagStandard52h13": "tag52_13_%05d.png"
                       }


def get_tag_img(tag_img_dir, tag_family, id, width=200, invert=False):
    image_path = join(tag_img_dir, tag_family, tag_family_template[tag_family] % id)
    tag_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    scaled_tag_img = cv2.resize(tag_image, (width, width), interpolation=cv2.INTER_NEAREST)
    if invert:
        scaled_tag_img = 255 - scaled_tag_img
    return scaled_tag_img


def save_tag_dict(tag_dict_path, tag_dict):
    print("save")


def read_tag_dict(tag_dict_path):
    print("read")

