import numpy as np


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                [rot_cos, -rot_sin, zeros],
                [rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                [zeros, rot_cos, -rot_sin],
                [zeros, rot_sin, rot_cos],
                [ones, zeros, zeros],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box3d(centers, dims, angles=None, axis=2):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    # corners = corners_nd(dims, origin=origin)
    corners_origin = np.array(
       [[ 0.5,  0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5,  0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5]]
    )
    corners_origin = np.expand_dims(corners_origin, 0).repeat(centers.shape[0], axis=0)

    corners = np.expand_dims(dims, 1) * corners_origin
    # print(corners.shape)

    # print("corners = \n", corners)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

####################################################################################


def compute_3d_corners_kitti(boxes7d):
    """
    Args:
        boxes7d (np.array): [N * 7]
    Returns:
        (np.array): [N * 3 * 8]
    """
    centers = boxes7d[:, :3]
    dims = boxes7d[:, 3:6]
    angles = boxes7d[:, 6]

    # return center_to_corner_box3d(centers, dims, angles).transpose(0,2,1)
    return center_to_corner_box3d(centers, dims, angles)

def compute_3d_cornors(x, y, z, dx, dy, dz, yaw, pose=None):

    R = np.array([[ np.cos(yaw), -np.sin(yaw), 0], 
                  [np.sin(yaw), np.cos(yaw), 0], 
                  [           0,           0, 1]])

    x_corners = [dx/2, -dx/2, -dx/2, dx/2,
                 dx/2, -dx/2, -dx/2, dx/2]

    y_corners = [-dy/2, -dy/2, dy/2, dy/2,
                 -dy/2, -dy/2, dy/2, dy/2]
    z_corners = [-dz/2,  -dz/2,  -dz/2,  -dz/2,
                 dz/2, dz/2, dz/2, dz/2]
                     
    xyz = np.vstack([x_corners, y_corners, z_corners])
    corners_3d_cam2 = np.zeros((4,8),dtype=np.float32)
    corners_3d_cam2[-1] = 1
    # print(xyz)
    corners_3d_cam2[:3] = np.dot(R, xyz)
    corners_3d_cam2[0,:] += x
    corners_3d_cam2[1,:] += y
    corners_3d_cam2[2,:] += z
    # print(corners_3d_cam2.shape)

    if pose is not None:
        pose = np.matrix(pose)
        corners_3d_cam2 = np.matmul(pose.I, corners_3d_cam2)

    return corners_3d_cam2[:3]

def convert_kitti_waymo(boxes7d):
    boxes7d_new = boxes7d[:, [0,1,2,4,3,5,6]]
    boxes7d_new[:, 6] = -boxes7d_new[:, 6] - np.pi/2.0
    return boxes7d_new


def vector_rotate(vec_in, angle):
    R = np.array([[ np.cos(angle), -np.sin(angle)], 
                  [ np.sin(angle),  np.cos(angle)]])
    return np.matmul(vec_in, R.T)




if __name__ == "__main__":
    boxes = np.array([[10,20,30,4,2,1,0]]).repeat(1000, axis=0)
    corners = compute_3d_corners_kitti(boxes)
    print(corners)
    print(corners.shape)

    vec = np.array([[1.1, 2.2],
                    [10.1, 20.2]])
    
    vec2 = vector_rotate(vec, np.pi * 2)
    print(vec2)

