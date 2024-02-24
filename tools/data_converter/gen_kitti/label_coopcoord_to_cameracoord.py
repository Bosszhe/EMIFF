import os
import numpy as np
import json
import errno
import math

from tqdm import tqdm
import os.path as osp
from .dataset_utils import InfFrame, VehFrame, VICFrame, Label_kitti
from .v2x_utils import Filter, RectFilter, id_cmp, id_to_str, get_trans, box_translation, get_3d_8points, get_xyzlwhy, range2box
from .gen_calib2kitti import convert_calib_v2x_to_kitti, get_cam_D_and_cam_K


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

def read_json(path_json):
    with open(path_json, "r") as load_f:
        my_json = json.load(load_f)
    return my_json


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_label(label):
    h = float(label["3d_dimensions"]["h"])
    w = float(label["3d_dimensions"]["w"])
    length = float(label["3d_dimensions"]["l"])
    x = float(label["3d_location"]["x"])
    y = float(label["3d_location"]["y"])
    z = float(label["3d_location"]["z"])
    rotation_y = float(label["rotation"])
    return h, w, length, x, y, z, rotation_y


def set_label(label, h, w, length, x, y, z, alpha, rotation_y):
    label["3d_dimensions"]["h"] = h
    label["3d_dimensions"]["w"] = w
    label["3d_dimensions"]["l"] = length
    label["3d_location"]["x"] = x
    label["3d_location"]["y"] = y
    label["3d_location"]["z"] = z
    label["alpha"] = alpha
    label["rotation_y"] = rotation_y


def normalize_angle(angle):
    # make angle in range [-0.5pi, 1.5pi]
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan


def get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    )
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    corners_3d_cam = r_velo2cam * corners_3d_lidar + t_velo2cam

    x0, z0 = corners_3d_cam[0, 0], corners_3d_cam[2, 0]
    x3, z3 = corners_3d_cam[0, 3], corners_3d_cam[2, 3]
    dx, dz = x0 - x3, z0 - z3
    yaw = math.atan2(-dz, dx)

    alpha = yaw - math.atan2(center_in_cam[0], center_in_cam[2])

    # add transfer
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi

    alpha_arctan = normalize_angle(alpha)

    return alpha_arctan, yaw


def convert_point(point, matrix):
    return matrix @ point


def get_lidar2cam(calib):
    r_velo2cam = np.array(calib["rotation"])
    t_velo2cam = np.array(calib["translation"])
    r_velo2cam = r_velo2cam.reshape(3, 3)
    t_velo2cam = t_velo2cam.reshape(3, 1)
    return r_velo2cam, t_velo2cam


def get_lidar2cam_2(calib_path):
    calib = load_json(calib_path)

    if "Tr_velo_to_cam" in calib.keys():
        velo2cam = np.array(calib["Tr_velo_to_cam"]).reshape(3, 4)
        r_velo2cam = velo2cam[:, :3]
        t_velo2cam = velo2cam[:, 3].reshape(3, 1)
    else:
        r_velo2cam = np.array(calib["rotation"])
        t_velo2cam = np.array(calib["translation"])
    return r_velo2cam, t_velo2cam



def build_path_to_info(prefix, data, sensortype="lidar"):
    path2info = {}
    if sensortype == "lidar":
        for elem in data:
            if elem["pointcloud_path"] == "":
                continue
            path = osp.join(prefix, elem["pointcloud_path"])
            path2info[path] = elem
    elif sensortype == "camera":
        for elem in data:
            if elem["image_path"] == "":
                continue
            path = osp.join(prefix, elem["image_path"])
            path2info[path] = elem
    return path2info

def get_trans(info):
    return info["translation"], info["rotation"]



def get_calibs(calib_path):
    calib = read_json(calib_path)
    if 'transform' in calib.keys():
        calib = calib['transform']
    rotation = calib['rotation']
    translation = calib['translation']
    return rotation, translation

def rev_matrix(rotation, translation):
    rotation = np.matrix(rotation)
    rev_R = rotation.I
    rev_R = np.array(rev_R)
    rev_T = - np.dot(rev_R, translation)
    return rev_R, rev_T

def inverse_matrix(R):
    R = np.matrix(R)
    rev_R = R.I
    rev_R = np.array(rev_R)
    return rev_R

def mul_matrix(rotation_1, translation_1, rotation_2, translation_2):
    rotation_1 = np.matrix(rotation_1)
    translation_1 = np.matrix(translation_1)
    rotation_2 = np.matrix(rotation_2)
    translation_2 = np.matrix(translation_2)

    rotation = rotation_2 * rotation_1
    translation = rotation_2 * translation_1 + translation_2
    rotation = np.array(rotation)
    translation = np.array(translation)

    return rotation, translation

def trans_lidar_i2v(inf_lidar2world_path, veh_lidar2novatel_path,
                    veh_novatel2world_path, system_error_offset=None):
    inf_lidar2world_r, inf_lidar2world_t = get_calibs(inf_lidar2world_path)
    if system_error_offset is not None:
        inf_lidar2world_t[0][0] = inf_lidar2world_t[0][0] + system_error_offset['delta_x']
        inf_lidar2world_t[1][0] = inf_lidar2world_t[1][0] + system_error_offset['delta_y']

    veh_novatel2world_r, veh_novatel2world_t = get_calibs(veh_novatel2world_path)
    veh_world2novatel_r, veh_world2novatel_t = rev_matrix(veh_novatel2world_r, veh_novatel2world_t)
    inf_lidar2novatel_r, inf_lidar2novatel_t = mul_matrix(inf_lidar2world_r, inf_lidar2world_t,
                                                          veh_world2novatel_r, veh_world2novatel_t)

    veh_lidar2novatel_r, veh_lidar2novatel_t = get_calibs(veh_lidar2novatel_path)
    veh_novatel2lidar_r, veh_novatel2lidar_t = rev_matrix(veh_lidar2novatel_r, veh_lidar2novatel_t)
    inf_lidar2lidar_r,  inf_lidar2lidar_t = mul_matrix(inf_lidar2novatel_r, inf_lidar2novatel_t,
                                                       veh_novatel2lidar_r, veh_novatel2lidar_t)

    return inf_lidar2lidar_r, inf_lidar2lidar_t



def get_novatel2world(path_novatel2world):
    novatel2world = read_json(path_novatel2world)
    rotation = novatel2world['rotation']
    translation = novatel2world['translation']
    return rotation, translation


def get_lidar2novatel(path_lidar2novatel):
    lidar2novatel = read_json(path_lidar2novatel)
    rotation = lidar2novatel['transform']['rotation']
    translation = lidar2novatel['transform']['translation']
    return rotation, translation


def trans_point(input_point, translation, rotation):
    input_point = np.array(input_point).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, 1) + np.array(translation).reshape(3, 1)
    output_point = output_point.reshape(1, 3).tolist()
    return output_point[0]

def trans_point_w2l(input_point, path_novatel2world, path_lidar2novatel):
    # world to novatel
    rotation, translation = get_novatel2world(path_novatel2world)
    new_rotation = inverse_matrix(rotation)
    new_translation = - np.dot(new_rotation, translation)
    point = trans_point(input_point, new_translation, new_rotation)

    # novatel to lidar
    rotation, translation = get_lidar2novatel(path_lidar2novatel)
    new_rotation = inverse_matrix(rotation)
    new_translation = - np.dot(new_rotation, translation)
    point = trans_point(point, new_translation, new_rotation)

    return point
    

def get_label_lidar_rotation(lidar_3d_8_points):
    """
    3D box in LiDAR coordinate system:

          4 -------- 5
         /|         /|
        7 -------- 6 .
        | |        | |
        . 0 -------- 1
        |/         |/
        3 -------- 2

        x: 3->0
        y: 1->0
        z: 0->4

        Args:
            lidar_3d_8_points: eight point list [[x,y,z],...]
        Returns:
            rotation_z: (-pi,pi) rad
    """
    x0, y0 = lidar_3d_8_points[0][0], lidar_3d_8_points[0][1]
    x3, y3 = lidar_3d_8_points[3][0], lidar_3d_8_points[3][1]
    dx, dy = x0 - x3, y0 - y3
    rotation_z = math.atan2(dy, dx)
    return rotation_z


def label_world2vlidar(sub_root, idx):
    path_input_label_file = os.path.join(sub_root, 'cooperative/label_world', idx + '.json')
    # path_output_label_dir = os.path.join(sub_root, 'cooperative/label/lidar')
    # if not os.path.exists(path_output_label_dir):
    #     os.makedirs(path_output_label_dir)
    # path_output_label_file = os.path.join(path_output_label_dir, idx + '.json')

    input_label_data = read_json(path_input_label_file)
    lidar_3d_list = []
    path_novatel2world = os.path.join(sub_root, 'vehicle-side/calib/novatel_to_world', idx + '.json')
    path_lidar2novatel = os.path.join(sub_root, 'vehicle-side/calib/lidar_to_novatel', idx + '.json')
    for label_world in input_label_data:
        world_8_points_old = label_world["world_8_points"]
        world_8_points = []
        for point in world_8_points_old:
            point_new = trans_point_w2l(point, path_novatel2world, path_lidar2novatel)
            world_8_points.append(point_new)

        lidar_3d_data = {}
        lidar_3d_data['type'] = label_world['type']
        lidar_3d_data['occluded_state'] = label_world['occluded_state']
        lidar_3d_data["truncated_state"] = label_world['truncated_state']
        lidar_3d_data['2d_box'] = label_world['2d_box']
        lidar_3d_data["3d_dimensions"] = label_world['3d_dimensions']
        lidar_3d_data["3d_location"] = {}
        lidar_3d_data["3d_location"]["x"] = (world_8_points[0][0] + world_8_points[2][0]) / 2
        lidar_3d_data["3d_location"]["y"] = (world_8_points[0][1] + world_8_points[2][1]) / 2
        lidar_3d_data["3d_location"]["z"] = (world_8_points[0][2] + world_8_points[4][2]) / 2
        lidar_3d_data["rotation"] = get_label_lidar_rotation(world_8_points)
        lidar_3d_list.append(lidar_3d_data)
    # write_json(path_output_label_file, lidar_3d_list)

    return lidar_3d_list


# ===================================================================# 
def gen_veh_lidar2veh_cam(source_root, target_root, label_type="lidar"):

    dair_v2x_c_root = source_root
    c_jsons_path = os.path.join(dair_v2x_c_root, 'cooperative/data_info.json')
    c_jsons = read_json(c_jsons_path)
    write_path = os.path.join(target_root, "label", label_type)
    mkdir_p(write_path)

    for c_json in tqdm(c_jsons):
        inf_idx = c_json['infrastructure_image_path'].split('/')[-1].replace('.jpg', '')
        inf_lidar2world_path = os.path.join(dair_v2x_c_root,
                                            'infrastructure-side/calib/virtuallidar_to_world/' + inf_idx + '.json')
        veh_idx = c_json['vehicle_image_path'].split('/')[-1].replace('.jpg', '')
        veh_lidar2novatel_path = os.path.join(dair_v2x_c_root,
                                              'vehicle-side/calib/lidar_to_novatel/' + veh_idx + '.json')
        veh_novatel2world_path = os.path.join(dair_v2x_c_root,
                                              'vehicle-side/calib/novatel_to_world/' + veh_idx + '.json')
        system_error_offset = c_json['system_error_offset']
        if system_error_offset == "":
            system_error_offset = None

        # inf_lidar2veh_lidar matrix
        # calib_lidar_i2v_r, calib_lidar_i2v_t = trans_lidar_i2v(inf_lidar2world_path, veh_lidar2novatel_path,
                                        #   veh_novatel2world_path, system_error_offset)


        c_json['calib_v_lidar2cam_path'] = os.path.join('vehicle-side/calib/lidar_to_camera', veh_idx + '.json')
        c_json['calib_v_cam_intrinsic_path'] = os.path.join('vehicle-side/calib/camera_intrinsic/', veh_idx + '.json')
        c_json['calib_lidar_i2v_path'] = os.path.join('cooperative/calib/lidar_i2v', veh_idx + '.json')


        

        calib_lidar2cam_path = c_json['calib_v_lidar2cam_path']
        calib_lidar2cam = read_json(os.path.join(dair_v2x_c_root, calib_lidar2cam_path))
        r_velo2cam, t_velo2cam = get_lidar2cam(calib_lidar2cam)
        Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))

        label_veh_lidar_list = label_world2vlidar(dair_v2x_c_root, veh_idx)

        for label in label_veh_lidar_list:
            h, w, l, x, y, z, yaw_lidar = get_label(label)
            z = z - h / 2
            bottom_center = [x, y, z]
            obj_size = [l, w, h]

            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(
                obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
            )
            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)

            set_label(label, h, w, l, cam_x, cam_y, cam_z, alpha, yaw)


        labels_path = os.path.join('label', 'lidar', veh_idx + '.json')
        write_path = os.path.join(target_root, labels_path)

        # print(target_root)

        
        with open(write_path, "w") as f:
            json.dump(label_veh_lidar_list, f)
            

# ===================================================================# 
# ===================================================================# 
# ===================================================================# 
# ===================================================================# 
# ===================================================================# 

def seq_label_coop2vlidar(sub_root, idx):
    path_input_label_file = os.path.join(sub_root, 'cooperative/label-lidar', idx + '.json')
    # path_output_label_dir = os.path.join(sub_root, 'cooperative/label/lidar')
    # if not os.path.exists(path_output_label_dir):
    #     os.makedirs(path_output_label_dir)
    # path_output_label_file = os.path.join(path_output_label_dir, idx + '.json')

    input_label_data = read_json(path_input_label_file)
    lidar_3d_list = []
    # path_novatel2world = os.path.join(sub_root, 'vehicle-side/calib/novatel_to_world', idx + '.json')
    # path_lidar2novatel = os.path.join(sub_root, 'vehicle-side/calib/lidar_to_novatel', idx + '.json')
    for label_world in input_label_data:

        lidar_3d_data = {}
        lidar_3d_data['type'] = label_world['type']
        lidar_3d_data['occluded_state'] = label_world['occluded_state']
        lidar_3d_data["truncated_state"] = label_world['truncated_state']
        lidar_3d_data['2d_box'] = label_world['2d_box']
        lidar_3d_data["3d_dimensions"] = label_world['3d_dimensions']
        lidar_3d_data["3d_location"] = label_world['3d_location']
        lidar_3d_data["rotation"] = label_world['rotation']
        lidar_3d_list.append(lidar_3d_data)
    # write_json(path_output_label_file, lidar_3d_list)

    return lidar_3d_list




def seq_label_coop2ilidar(sub_root, veh_idx, inf_idx, system_error_offset):
    path_input_label_file = os.path.join(sub_root, 'cooperative/label-lidar', veh_idx + '.json')
    dair_v2x_c_root = '/home/wangz/wangzhe21/DAIR-V2X-BEV/data/DAIR_V2X_Seq/SPD/cooperative-vehicle-infrastructure'
    inf_lidar2world_path = os.path.join(dair_v2x_c_root,
                                            'infrastructure-side/calib/virtuallidar_to_world/' + inf_idx + '.json')
    veh_lidar2novatel_path = os.path.join(dair_v2x_c_root,
                                              'vehicle-side/calib/lidar_to_novatel/' + veh_idx + '.json')
    veh_novatel2world_path = os.path.join(dair_v2x_c_root,
                                              'vehicle-side/calib/novatel_to_world/' + veh_idx + '.json')
    
    r_inf_lidar2world, t_inf_lidar2world = get_calibs(inf_lidar2world_path)
    veh_novatel2world_r, veh_novatel2world_t = get_calibs(veh_novatel2world_path)
    veh_world2novatel_r, veh_world2novatel_t = rev_matrix(veh_novatel2world_r, veh_novatel2world_t)
    veh_lidar2novatel_r, veh_lidar2novatel_t = get_calibs(veh_lidar2novatel_path)
    veh_novatel2lidar_r, veh_novatel2lidar_t = rev_matrix(veh_lidar2novatel_r, veh_lidar2novatel_t)
    r_world2veh_lidar,  t_world2veh_lidar = mul_matrix(veh_world2novatel_r, veh_world2novatel_t,
                                                    veh_novatel2lidar_r, veh_novatel2lidar_t)
    r_inf_lidar2veh_lidar,  t_inf_lidar2veh_lidar = mul_matrix(r_inf_lidar2world, t_inf_lidar2world,
                                                    r_world2veh_lidar, t_world2veh_lidar)
    t_inf_lidar2veh_lidar = t_inf_lidar2veh_lidar + np.array([system_error_offset['delta_x'], system_error_offset['delta_y'], 0]).reshape(3, 1)

    r_veh_lidar2inf_lidar, t_veh_lidar2inf_lidar = rev_matrix(r_inf_lidar2veh_lidar, t_inf_lidar2veh_lidar)


    input_label_data = read_json(path_input_label_file)
    lidar_3d_list = []

    for label_world in input_label_data:
        h, w, l, x, y, z, yaw_lidar = get_label(label_world)
        obj_size=np.array([l,w,h])
        center_lidar = np.array([x,y,z])
        world_8_points_old = get_3d_8points(obj_size, yaw_lidar, center_lidar)
    
        world_8_points = []
        for point in world_8_points_old:
            point_new = trans_point(point, t_veh_lidar2inf_lidar, r_veh_lidar2inf_lidar)
            world_8_points.append(point_new)

        lidar_3d_data = {}
        lidar_3d_data['type'] = label_world['type']
        lidar_3d_data['occluded_state'] = label_world['occluded_state']
        lidar_3d_data["truncated_state"] = label_world['truncated_state']
        lidar_3d_data['2d_box'] = label_world['2d_box']
        lidar_3d_data["3d_dimensions"] = label_world['3d_dimensions']
        lidar_3d_data["3d_location"] = {}
        lidar_3d_data["3d_location"]["x"] = (world_8_points[0][0] + world_8_points[2][0]) / 2
        lidar_3d_data["3d_location"]["y"] = (world_8_points[0][1] + world_8_points[2][1]) / 2
        lidar_3d_data["3d_location"]["z"] = (world_8_points[0][2] + world_8_points[4][2]) / 2
        lidar_3d_data["rotation"] = get_label_lidar_rotation(world_8_points)
        lidar_3d_list.append(lidar_3d_data)
    # write_json(path_output_label_file, lidar_3d_list)

    return lidar_3d_list


# ===================================================================# 
def seq_gen_veh_lidar2veh_cam(source_root, target_root, label_type="lidar"):

    dair_v2x_c_root = source_root
    c_jsons_path = os.path.join(dair_v2x_c_root, 'cooperative/data_info.json')
    c_jsons = read_json(c_jsons_path)
    write_path = os.path.join(target_root, "label", label_type)
    mkdir_p(write_path)

    for c_json in tqdm(c_jsons):
        inf_idx = c_json['infrastructure_frame']
        inf_lidar2world_path = os.path.join(dair_v2x_c_root,
                                            'infrastructure-side/calib/virtuallidar_to_world/' + inf_idx + '.json')
        veh_idx = c_json['vehicle_frame']
        veh_lidar2novatel_path = os.path.join(dair_v2x_c_root,
                                              'vehicle-side/calib/lidar_to_novatel/' + veh_idx + '.json')
        veh_novatel2world_path = os.path.join(dair_v2x_c_root,
                                              'vehicle-side/calib/novatel_to_world/' + veh_idx + '.json')
        system_error_offset = c_json['system_error_offset']
        if system_error_offset == "":
            system_error_offset = None

        # inf_lidar2veh_lidar matrix
        # calib_lidar_i2v_r, calib_lidar_i2v_t = trans_lidar_i2v(inf_lidar2world_path, veh_lidar2novatel_path,
                                        #   veh_novatel2world_path, system_error_offset)


        c_json['calib_v_lidar2cam_path'] = os.path.join('vehicle-side/calib/lidar_to_camera', veh_idx + '.json')
        # c_json['calib_v_cam_intrinsic_path'] = os.path.join('vehicle-side/calib/camera_intrinsic/', veh_idx + '.json')
        # c_json['calib_lidar_i2v_path'] = os.path.join('cooperative/calib/lidar_i2v', veh_idx + '.json')


        

        calib_lidar2cam_path = c_json['calib_v_lidar2cam_path']
        calib_lidar2cam = read_json(os.path.join(dair_v2x_c_root, calib_lidar2cam_path))
        r_velo2cam, t_velo2cam = get_lidar2cam(calib_lidar2cam)
        Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))

        label_veh_lidar_list = seq_label_coop2vlidar(dair_v2x_c_root, veh_idx)

        for label in label_veh_lidar_list:
            h, w, l, x, y, z, yaw_lidar = get_label(label)
            z = z - h / 2
            bottom_center = [x, y, z]
            obj_size = [l, w, h]

            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(
                obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
            )
            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)

            set_label(label, h, w, l, cam_x, cam_y, cam_z, alpha, yaw)


        labels_path = os.path.join('label', 'lidar', veh_idx + '.json')
        write_path = os.path.join(target_root, labels_path)

        # print(target_root)

        
        with open(write_path, "w") as f:
            json.dump(label_veh_lidar_list, f)

# ===================================================================# 
def seq_gen_veh_lidar2inf_cam(source_root, target_root, label_type="lidar"):

    dair_v2x_c_root = source_root
    c_jsons_path = os.path.join(dair_v2x_c_root, 'cooperative/data_info.json')
    c_jsons = read_json(c_jsons_path)
    write_path = os.path.join(target_root, "label", label_type)
    mkdir_p(write_path)

    for c_json in tqdm(c_jsons):
        inf_idx = c_json['infrastructure_frame']
        veh_idx = c_json['vehicle_frame']
        system_error_offset = c_json['system_error_offset']
        # if system_error_offset == "":
        #     system_error_offset = None

        c_json['calib_i_lidar2cam_path'] = os.path.join('infrastructure-side/calib/virtuallidar_to_camera', inf_idx + '.json')

        calib_lidar2cam_path = c_json['calib_i_lidar2cam_path']
        calib_lidar2cam = read_json(os.path.join(dair_v2x_c_root, calib_lidar2cam_path))
        r_velo2cam, t_velo2cam = get_lidar2cam(calib_lidar2cam)
        Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))

        label_veh_lidar_list = seq_label_coop2ilidar(dair_v2x_c_root, veh_idx, inf_idx,system_error_offset)

        for label in label_veh_lidar_list:
            h, w, l, x, y, z, yaw_lidar = get_label(label)
            z = z - h / 2
            bottom_center = [x, y, z]
            obj_size = [l, w, h]

            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(
                obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
            )
            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)

            set_label(label, h, w, l, cam_x, cam_y, cam_z, alpha, yaw)


        labels_path = os.path.join('label', 'lidar', inf_idx + '.json')
        write_path = os.path.join(target_root, labels_path)

        # print(target_root)

        
        with open(write_path, "w") as f:
            json.dump(label_veh_lidar_list, f)


def seq_gen_calib2kitti_coop(source_root, target_root, label_type="lidar"):

    path_calib = os.path.join(target_root, "training/calib")
    mkdir_p(path_calib)

    dair_v2x_c_root = source_root
    c_jsons_path = os.path.join(dair_v2x_c_root, 'cooperative/data_info.json')
    c_jsons = read_json(c_jsons_path)


    for c_json in tqdm(c_jsons):
        inf_idx = c_json['infrastructure_frame'].split('/')[-1].replace('.jpg', '')
        inf_lidar2world_path = os.path.join(dair_v2x_c_root,
                                            'infrastructure-side/calib/virtuallidar_to_world/' + inf_idx + '.json')
        veh_idx = c_json['vehicle_frame'].split('/')[-1].replace('.jpg', '')
        veh_lidar2novatel_path = os.path.join(dair_v2x_c_root,
                                              'vehicle-side/calib/lidar_to_novatel/' + veh_idx + '.json')
        veh_novatel2world_path = os.path.join(dair_v2x_c_root,
                                              'vehicle-side/calib/novatel_to_world/' + veh_idx + '.json')
        system_error_offset = c_json['system_error_offset']
        if system_error_offset == "":
            system_error_offset = None

        r_inf_lidar2world, t_inf_lidar2world = get_calibs(inf_lidar2world_path)
        r_world2inf_lidar, t_world2inf_lidar = rev_matrix(r_inf_lidar2world, t_inf_lidar2world)

        veh_novatel2world_r, veh_novatel2world_t = get_calibs(veh_novatel2world_path)
        veh_world2novatel_r, veh_world2novatel_t = rev_matrix(veh_novatel2world_r, veh_novatel2world_t)
        veh_lidar2novatel_r, veh_lidar2novatel_t = get_calibs(veh_lidar2novatel_path)
        veh_novatel2lidar_r, veh_novatel2lidar_t = rev_matrix(veh_lidar2novatel_r, veh_lidar2novatel_t)
        r_world2veh_lidar,  t_world2veh_lidar = mul_matrix(veh_world2novatel_r, veh_world2novatel_t,
                                                       veh_novatel2lidar_r, veh_novatel2lidar_t)

        r_inf_lidar2veh_lidar,  t_inf_lidar2veh_lidar = mul_matrix(r_inf_lidar2world, t_inf_lidar2world,
                                                       r_world2veh_lidar, t_world2veh_lidar)

        t_inf_lidar2veh_lidar = t_inf_lidar2veh_lidar + np.array([system_error_offset['delta_x'], system_error_offset['delta_y'], 0]).reshape(3, 1)





        c_json['calib_v_lidar2cam_path'] = os.path.join('vehicle-side/calib/lidar_to_camera', veh_idx + '.json')
        c_json['calib_i_lidar2cam_path'] = os.path.join('infrastructure-side/calib/virtuallidar_to_camera', inf_idx + '.json')



        c_json['calib_v_cam_intrinsic_path'] = os.path.join('vehicle-side/calib/camera_intrinsic/', veh_idx + '.json')
        c_json['calib_i_cam_intrinsic_path'] = os.path.join('infrastructure-side/calib/camera_intrinsic/', inf_idx + '.json')
        

        # fetch transformation matrixs
        # intrinstic 3*3
        veh_cam_D, veh_cam_K = get_cam_D_and_cam_K(os.path.join(dair_v2x_c_root,c_json['calib_v_cam_intrinsic_path']))
        inf_cam_D, inf_cam_K = get_cam_D_and_cam_K(os.path.join(dair_v2x_c_root,c_json['calib_i_cam_intrinsic_path']))
    
        calib_veh_lidar2cam = read_json(os.path.join(dair_v2x_c_root, c_json['calib_v_lidar2cam_path']))
        r_veh_lidar2cam, t_veh_lidar2cam = get_lidar2cam(calib_veh_lidar2cam)

        calib_inf_lidar2cam = read_json(os.path.join(dair_v2x_c_root, c_json['calib_i_lidar2cam_path']))
        r_inf_lidar2cam, t_inf_lidar2cam = get_lidar2cam(calib_inf_lidar2cam)
        
        txt_name = veh_idx+ ".txt"
        txt_path = os.path.join(path_calib, txt_name)

        
        P2, veh_Tr_velo_to_cam = convert_calib_v2x_to_kitti(veh_cam_D, veh_cam_K, t_veh_lidar2cam, r_veh_lidar2cam)
        P3, inf_Tr_velo_to_cam = convert_calib_v2x_to_kitti(inf_cam_D, inf_cam_K, t_inf_lidar2cam, r_inf_lidar2cam)
        _, inf_lidar2veh_lidar = convert_calib_v2x_to_kitti(veh_cam_D, veh_cam_K, t_inf_lidar2veh_lidar, r_inf_lidar2veh_lidar)
        _, world2veh_lidar = convert_calib_v2x_to_kitti(veh_cam_D, veh_cam_K, t_world2veh_lidar, r_world2veh_lidar)
        _, world2inf_lidar = convert_calib_v2x_to_kitti(veh_cam_D, veh_cam_K, t_world2inf_lidar, r_world2inf_lidar)

        
        str_P2 = "P2: "
        str_P3 = "P3: "
        str_veh_Tr_velo_to_cam = "veh_Tr_velo_to_cam: "
        str_inf_Tr_velo_to_cam = "inf_Tr_velo_to_cam: "
        str_inf_lidar2veh_lidar = "inf_lidar2veh_lidar: "
        str_world2veh_lidar = "world2veh_lidar: "
        str_world2inf_lidar = "world2inf_lidar: "
        
        for ii in range(11):
            str_P2 = str_P2 + str(P2[ii]) + " "
            str_P3 = str_P3 + str(P3[ii]) + " "
            str_veh_Tr_velo_to_cam = str_veh_Tr_velo_to_cam + str(veh_Tr_velo_to_cam[ii]) + " "
            str_inf_Tr_velo_to_cam = str_inf_Tr_velo_to_cam + str(inf_Tr_velo_to_cam[ii]) + " "
            str_inf_lidar2veh_lidar = str_inf_lidar2veh_lidar + str(inf_lidar2veh_lidar[ii]) + " "
            str_world2veh_lidar = str_world2veh_lidar + str(world2veh_lidar[ii]) + " "
            str_world2inf_lidar = str_world2inf_lidar + str(world2inf_lidar[ii]) + " "
        str_P2 = str_P2 + str(P2[11])
        str_P3 = str_P3 + str(P3[11])
        str_veh_Tr_velo_to_cam = str_veh_Tr_velo_to_cam + str(veh_Tr_velo_to_cam[11])
        str_inf_Tr_velo_to_cam = str_inf_Tr_velo_to_cam + str(inf_Tr_velo_to_cam[11])
        str_inf_lidar2veh_lidar = str_inf_lidar2veh_lidar + str(inf_lidar2veh_lidar[11])
        str_world2veh_lidar = str_world2veh_lidar + str(world2veh_lidar[11])
        str_world2inf_lidar = str_world2inf_lidar + str(world2inf_lidar[11])

        str_P0 = str_P2
        str_P1 = str_P2
        str_R0_rect = "R0_rect: 1 0 0 0 1 0 0 0 1"
        str_Tr_imu_to_velo = str_veh_Tr_velo_to_cam

        with open(txt_path, "w") as fp:
            gt_line = (
                str_P0
                + "\n"
                + str_P1
                + "\n"
                + str_P2
                + "\n"
                + str_P3
                + "\n"
                + str_R0_rect
                + "\n"
                + str_veh_Tr_velo_to_cam
                + "\n"
                + str_inf_Tr_velo_to_cam
                + "\n"
                + str_inf_lidar2veh_lidar
                + "\n"
                + str_Tr_imu_to_velo
                + "\n"
                + str_world2veh_lidar
                + "\n"
                + str_world2inf_lidar
            )
            fp.write(gt_line)



