#!/usr/bin/python3

# from sort_3d import Sort
# from pc3t import Sort
from unittest.case import _BaseTestCaseContext
from data_utils import *
from publish_utils import *
import glob
# from sort_3d import Sort
from sort_3d_wh import Sort
# from sort_3d_ukf import Sort
from sort_2d import Sort2d, associate_detections_to_trackers

from lidar_camera_sync import LidarCameraSync,cmap
import pickle as pkl
import json
import numpy as np
import copy
np.random.seed(100)

ROS_RATE = 6
SCORE_THRESH = 0.2


def draw_box_3d(image, corners, c=(255, 0, 255), same_color=False, text = None):
  face_idx = [[0,1,5,4],
              [1,2,6,5],
              [3,0,4,7],
              [2,3,7,6]]
  thickness = 1
  corners = corners.astype(np.int32)
  place = (int(corners[:,0].min()), int(corners[:,1].min())-5)
  cv2.putText(img, text =  str(text), org = place,  fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.0, color = c, thickness = 1)

  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      try:
        cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
            (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, thickness, lineType=cv2.LINE_AA)
      except:
        pass
    if ind_f == 0:
      try:
        cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                 (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
        cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                 (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
      except:
        pass
    # top_idx = [0, 1, 2, 3]
  return image

def comput_corners_3d(dim, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners).transpose(1, 0)
  return corners_3d



def compute_box_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  corners_3d = comput_corners_3d(dim, rotation_y)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)
  return corners_3d


def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  # import pdb; pdb.set_trace()
  return pts_2d

def add_3d_detection(
  img, dets, calib, 
  vis_thresh=SCORE_THRESH, img_id='det'):

  for item in dets:
    if item['score'] > vis_thresh \
      and 'dim' in item and 'loc' in item and 'rot_y' in item:

      cl = tango_color_dark[int(item['class'])].tolist()
      dim = item['dim']
      loc = item['loc']
      rot_y = item['rot_y']
      if loc[2] > 1:
        box_3d = compute_box_3d(dim, loc, rot_y)
        box_2d = project_to_image(box_3d, calib)
        img= draw_box_3d(
          img, box_2d.astype(np.int32), cl, 
          same_color=False, text = np.around(item['score'],3 ))


rp_anos = "/home/wanghao/Desktop/projects/CP_TRT/CenterFusion/data/torch_results/save_annos_nuscenes.json"

result_root_pattern = "/home/wanghao/Desktop/projects/CP_TRT/CenterFusion/results/calib{}.bin.txt"
img_root_pattern = "/home/wanghao/Desktop/projects/CP_TRT/CenterFusion/data/cf_predata/img{}.bin"



tango_color_dark =np.array( [
  [114, 159, 207], #	Sky Blue 1
  [196, 160,   0], #	Butter 3
  [ 78, 154,   6], #	Chameleon 3
  [206,  92,   0], #	Orange 3
  [164,   0,   0], #	Scarlet Red 3
  [ 32,  74, 135], #	Sky Blue 3
  [ 92,  53, 102], #	Plum 3
  [143,  89,   2], #	Chocolate 3
  [ 85,  87,  83], #	Aluminium 5
  [186, 189, 182], #	Aluminium 3

  [0,0, 182], #	blue 

], dtype=np.int16)

    
def tlbr2yxhw(bboxs):
    hw = bboxs[:,2:4] - bboxs[:,:2]
    yx = (bboxs[:,2:4] + bboxs[:,:2])/2
    return np.concatenate([yx,hw],axis = -1)

def yxhw2tlbr(bboxs):
    tl = bboxs[:,:2] - bboxs[:,2:4]/2
    br = bboxs[:,:2] + bboxs[:,2:4]/2
    return np.concatenate([tl, br], axis= -1)

def get_veh_to_global(frame_name) :
    data = pickle.load(open(frame_name, "rb"))
    RT = data['veh_to_global'].reshape(4,4)
    return RT


def nusc2waymo_box(box3d_preds, rt):
    box3d_preds[:,:3] = box3d_preds[:,:3].dot(rt[:3,:3].T) + rt[:3,-1]
    box3d_preds[:,3:6] = box3d_preds[:,[3,5,4]]
    box3d_preds[:,6] = -box3d_preds[:,6] - np.pi/2
    return box3d_preds

def nusc2waymo(bbox, trans):
    # ctr, dim, rot  : 7-dof
    bbox[:,6] = -1 * bbox[:,6] - np.pi/2
    bbox[:, :3] = bbox[:,:3].dot(trans.T)
    bbox[:,3:6]= bbox[:,[5,4,3]]
    return bbox

def loop_add(x, v, size):
  x += v
  return x%size

if  __name__ == "__main__":

    frame = 0
    bridge = CvBridge()
    rospy.init_node('nusc_node',anonymous=True)
    pcl_pub = rospy.Publisher('nusc_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('nusc_ego_car',Marker, queue_size=10)
    box3d_pub = rospy.Publisher('nusc_3dbox',MarkerArray, queue_size=10)
    image_pub =rospy.Publisher("nusc_image", Image, queue_size = 10) 

    annos=pkl.load(open(rp_anos,'rb'))
    frame_nums = 1000

    calib =  LidarCameraSync(np.zeros(16), np.zeros(9))
    nusc2waymo_trans = np.array(
        [ [0,0,1,0], [-1,0,0,0], [0,-1,0,0], [0,0,0,1],]
    )
    
    intrinsics = np.array([[1.2528e+03, 0.0000e+00, 8.2659e+02],
                           [0.0000e+00, 1.2528e+03, 4.6998e+02],
                           [0.0000e+00, 0.0000e+00, 1.0000e+00]])
    calib.set_proj_param(nusc2waymo_trans.T, intrinsics)
    rate = rospy.Rate(ROS_RATE)

    ################################################################################
    while not rospy.is_shutdown():
        k = frame * 6 + 1
        anno = annos[ k]

        img = np.fromfile(img_root_pattern.format(frame), dtype=np.float32).reshape(3,448,800).transpose(1,2,0)
        img = (img-img.min())/(img.max() - img.min()) * 255
        img = cv2.resize(img.astype(np.uint8),(1600,900))
        preds = np.loadtxt(result_root_pattern.format(frame))
        if len(preds.shape)==1:
          frame = loop_add(frame, 1, frame_nums); continue
        #x, y, z, dx, dy, dz, v1, v2, rot, score, cls

        dis_keep_mask = np.logical_and(np.logical_and(preds[:,0]< 150, preds[:,0]> -150), np.logical_and(preds[:,2]< 150, preds[:,2]> -150))
        calib.set_proj_param(extrinsic=nusc2waymo_trans.T, intrinsic= anno['meta'][1.0]['calib'].squeeze().numpy()[:3,:3])

        score_keep_mask = preds[:,9] > SCORE_THRESH
        box_idxs = np.where(np.logical_and(dis_keep_mask, score_keep_mask))[0]
        if not len(box_idxs):
          frame = loop_add(frame, 1, frame_nums); continue

        dim = preds[box_idxs,3:6]
        ctr = preds[box_idxs,:3]
        # ctr[:,1] -= dim[:,2]
        rot = preds[box_idxs,8]
        scores = preds[box_idxs,9]
        scores = np.around (scores, 3)
        cls = preds[box_idxs,10].astype(np.int8)

        bboxs = np.concatenate([ctr,dim,rot.reshape(-1,1)], axis=-1)
        # bboxs = nusc2waymo_box(bboxs,nusc2waymo_trans)

        calib.proj_box_to_image(bboxs, img, color = tango_color_dark[cls], rand_color=False, all_in = False, show  = True, format_2d = False, 
        texts = cls)

        publish_camera(image_pub, bridge, img, borders_2d_cam2s=None) 

        publish_point_cloud(pcl_pub, bboxs[:,:3])
        
        publish_ego_car(ego_pub)


        
        corner_3d_velos = []
        for boxes in bboxs:
            # corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
            corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
            corner_3d_velos.append(np.array(corner_3d_velo).T)
        print("num boxes ", len(scores))
        publish_3dbox(box3d_pub, corner_3d_velos, texts = scores, types=cls, track_color=False, Lifetime=1. / ROS_RATE)

        rospy.loginfo("nusc published")
        rate.sleep()

        frame = loop_add(frame, 1, frame_nums);

