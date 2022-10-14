#!/usr/bin/python3
# -*- coding:utf-8 -*-

import cv2
import numpy as np

from lidar_loader import LidarDataLoader
from image_loader import ImageDataLoader
from box3d_loader import Box3dDataLoader
from poses_loader import PoseLoader
from lidar_camera_sync import LidarCameraSync
from box_utils import convert_kitti_waymo


class CVPublisher(object):
    def __init__(self,
                 cv_wait    = 10,
                 img_loader  = None,
                 box_loader  = None,
                 pc_loader = None,
                 extrinsic_path = None,
                 intrinsic_path = None,
                 ):
        self.cv_wait    = cv_wait
        self.img_loader  = img_loader
        self.box_loader  = box_loader
        self.pc_loader   = pc_loader
        self.intrinsic_path = intrinsic_path
        self.extrinsic_path = extrinsic_path
        self.load_intrinsic()
        self.load_extrinsic()
        self.lidarCameraSync = LidarCameraSync(self.extrinsic, self.intrinsic)
        # self.vehicle_to_image = get_image_transform(self.extrinsic, self.intrinsic)
        

    def load_intrinsic(self):
        # intrinsic: numpy (9)  [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        intrinsic = np.loadtxt(self.intrinsic_path)
        self.intrinsic = np.array([[intrinsic[0], 0, intrinsic[2]],
                                   [0, intrinsic[1], intrinsic[3]],
                                   [0, 0,                       1]])


    def load_extrinsic(self):
        extrinsic = np.loadtxt(self.extrinsic_path).reshape(4,4)
        # Swap the axes around
        axes_transformation = np.array([
            [0,-1,0,0],
            [0,0,-1,0],
            [1,0,0,0],
            [0,0,0,1]])
        self.extrinsic = np.matmul(axes_transformation, np.linalg.inv(extrinsic))


    def Process(self):
        frame_id = 0
        while True:
            image = self.img_loader.load_idx(frame_id)
            if image is None:
                return 0

            if self.box_loader is not None:
                box_infos = self.box_loader.load_idx(frame_id)
                if box_infos is None:
                    return 

            if self.pc_loader is not None:
                point_cloud = self.pc_loader.load_idx(frame_id)
                if point_cloud is None:
                    return 0
                image,_,_ = self.lidarCameraSync.proj_point_to_image(point_cloud, image)

            box_infos.boxes3d = convert_kitti_waymo(box_infos.boxes3d)
            # image = self.lidarCameraSync.proj_box_to_image(box_infos.boxes3d, image)
            image = self.lidarCameraSync.proj_track_info_to_image(box_infos, image)

            # image = cv2.resize(image, (960, 640))
            cv2.imshow("image", image)
            cv2.waitKey(self.cv_wait)

            seq = self.extrinsic_path.split("/")[-1].split("_")[1]
            cv2.imwrite("/mnt/data/waymo_opensets/val/image_res/seq_%s_frame_%d.jpg"%(seq, frame_id), image)

            frame_id += 1
            # print("published [%d]"%(frame_id))


if __name__ == "__main__":
    import glob

    # annos_path = "/mnt/data/waymo_opensets/val/annos_txt/"
    image_path = "/mnt/data/waymo_opensets/val/image/"
    annos_path = "/home/zhanghao/code/GL/trackingwithvelo/py/data/output/seq_6_30/22521_stablelose4_velo52_angle/txt/"
    lidar_path = "/mnt/data/waymo_opensets/val/lidar/"
    cam0_matrix_path = "/mnt/data/waymo_opensets/val/cam0_matrix/"

    tokens = []
    select_seq = [6, 10, 19, 21]
    # select_seq = [21]        

    for ii in select_seq:
        seq_ii_num = len(glob.glob(image_path + "/seq_%d_frame_*"%ii))
        tokens = ["seq_%d_frame_"%ii + str(jj) for jj in range(seq_ii_num)]
        print(tokens)

        anno_files  = [annos_path + tk + ".txt" for tk in tokens]
        image_files = [image_path + tk + ".jpg" for tk in tokens]
        lidar_files = [lidar_path + tk + ".pkl" for tk in tokens]

        lidar_generator = LidarDataLoader(lidar_files)
        annos_generator = Box3dDataLoader(anno_files, filt_thres=0.1, is_track=True)
        image_generator = ImageDataLoader(image_files)

        cvPublisher = CVPublisher(1, 
                                image_generator, 
                                annos_generator, 
                                lidar_generator, 
                                cam0_matrix_path + "/seq_%d_cam0_extrinsic.txt"%ii,
                                cam0_matrix_path + "/seq_%d_cam0_intrinsic.txt"%ii)
        cvPublisher.Process()



