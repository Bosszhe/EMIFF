import os
import numpy as np
import json
import errno
import math

from tqdm import tqdm
import os.path as osp
from .dataset_utils import InfFrame, VehFrame, VICFrame, Label_kitti, get_cam_calib_intrinsic, get_lidar2cam
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



def gen_lidar2cam_i(source_root, target_root, label_type="lidar"):
    write_path = os.path.join(target_root, "label", label_type)
    mkdir_p(write_path)
    
    path = 'data/DAIR-V2X/cooperative-vehicle-infrastructure'
    frame_pairs = read_json(os.path.join(path, "cooperative/data_info.json"))
    sensortype = 'camera'
    
    inf_path2info = build_path_to_info(
            "infrastructure-side",
            read_json(osp.join(path, "infrastructure-side/data_info.json")),
            sensortype,
        )
    veh_path2info = build_path_to_info(
            "vehicle-side",
            read_json(osp.join(path, "vehicle-side/data_info.json")),
            sensortype,
        )

    data = []
    inf_frames = {}
    veh_frames = {}

    for elem in tqdm(frame_pairs):
        if sensortype == "lidar":
            inf_frame = inf_path2info[elem["infrastructure_pointcloud_path"]]
            veh_frame = veh_path2info[elem["vehicle_pointcloud_path"]]
        elif sensortype == "camera":
            inf_frame = inf_path2info[elem["infrastructure_image_path"]]
            veh_frame = veh_path2info[elem["vehicle_image_path"]]
            
        inf_frame = InfFrame(path + "/infrastructure-side/", inf_frame)
        veh_frame = VehFrame(path + "/vehicle-side/", veh_frame)
        
        if not inf_frame["batch_id"] in inf_frames:
                inf_frames[inf_frame["batch_id"]] = [inf_frame]
        else:
            inf_frames[inf_frame["batch_id"]].append(inf_frame)
        if not veh_frame["batch_id"] in veh_frames:
            veh_frames[veh_frame["batch_id"]] = [veh_frame]
        else:
            veh_frames[veh_frame["batch_id"]].append(veh_frame)
        
        vic_frame = VICFrame(path, elem, veh_frame, inf_frame, 0)
        

        # if extended_range is not None:
        extended_range = range2box(np.array([0, -39.68, -3, 92.16, 39.68, 1]))
        trans = vic_frame.transform(from_coord="Vehicle_lidar", to_coord="World")
        filt_world = RectFilter(trans(extended_range)[0])
        label = Label_kitti(osp.join(path, elem["cooperative_label_path"]), filt_world)
        label_no_filt = Label_kitti(osp.join(path, elem["cooperative_label_path"]), None)
        
        print(label_no_filt['boxes_3d'].shape[0],label['boxes_3d'].shape[0])
        # from IPython import embed
        # embed(header='xxx')

        
        # label = Label_kitti(osp.join(path, elem["cooperative_label_path"]), None)
        
        trans_1 = vic_frame.transform("World", "Vehicle_lidar")
        label["veh_boxes_3d"] = trans_1(label["boxes_3d"])
        trans_2 = vic_frame.transform("World", "Infrastructure_lidar")
        label["inf_boxes_3d"] = trans_2(label["boxes_3d"])

        trans_3 = vic_frame.transform("World", "Infrastructure_lidar_nodelta")
        label["inf_boxes_3d_nodelta"] = trans_3(label["boxes_3d"])

        
        # labels_path = veh_frame["label_" + label_type + "_std_path"]
        labels_path = inf_frame["label_" + label_type + "_std_path"]
        
        # r_velo2cam, t_velo2cam = get_lidar2cam_2(
        #     osp.join(path, "vehicle-side", veh_frame["calib_lidar_to_camera_path"])
        # )
        
        # from IPython import embed
        # embed(header='inf')
        r_velo2cam, t_velo2cam = get_lidar2cam_2(
            osp.join(path, "infrastructure-side", inf_frame["calib_virtuallidar_to_camera_path"])
        )
        Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))
        



        
        labels_new = list()
        for i in range(len(label['boxes_3d'])):
            label_new = dict()
            label_new['3d_dimensions'] = dict()
            label_new['3d_location'] = dict()
            
            # gt_box = label['veh_boxes_3d'][i]
            gt_box = label['inf_boxes_3d'][i]
            label_new['type'] = label['types'][i]
            label_new['occluded_state'] = label['occluded_states'][i]
            label_new['truncated_state'] = label['truncated_states'][i]
            label_new['alpha'] = label['alphas'][i]
            label_new['2d_box'] = label['2d_boxes'][i]
                
                
            x, y, z, l, w, h, yaw_lidar = get_xyzlwhy(gt_box)
            z = z - h / 2
            bottom_center = [x, y, z]
            obj_size = [l, w, h]
            
            label_new['rotation'] = yaw_lidar
            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(
                obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
            )
            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)
            
            

            set_label(label_new, h, w, l, cam_x, cam_y, cam_z, alpha, yaw)
            
            labels_new.append(label_new)
            
        labels_path = labels_path.replace("virtuallidar", "lidar")
        write_path = os.path.join(target_root, labels_path)
        
        if os.path.exists(labels_path):
            print(labels_path)
        
        
        with open(write_path, "w") as f:
            json.dump(labels_new, f)
            

        
    # ===================================================================# 
def gen_lidar2cam(source_root, target_root, label_type="lidar"):
    path_data_info = os.path.join(source_root, "data_info.json")
    data_info = read_json(path_data_info)
    write_path = os.path.join(target_root, "label", label_type)
    mkdir_p(write_path)

    for data in tqdm(data_info):
        if "calib_virtuallidar_to_camera_path" in data.keys():
            calib_lidar2cam_path = data["calib_virtuallidar_to_camera_path"]
        else:
            calib_lidar2cam_path = data["calib_lidar_to_camera_path"]
        calib_lidar2cam = read_json(os.path.join(source_root, calib_lidar2cam_path))
        r_velo2cam, t_velo2cam = get_lidar2cam(calib_lidar2cam)
        Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))

        labels_path = data["label_" + label_type + "_std_path"]
        labels = read_json(os.path.join(source_root, labels_path))
        
        for label in labels:
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

        labels_path = labels_path.replace("virtuallidar", "lidar")
        write_path = os.path.join(target_root, labels_path)

        
        with open(write_path, "w") as f:
            json.dump(labels, f)
    
    
def gen_lidar2cam_v(source_root, target_root, label_type="lidar"):
    write_path = os.path.join(target_root, "label", label_type)
    mkdir_p(write_path)
    
    path = 'data/DAIR-V2X/cooperative-vehicle-infrastructure'
    frame_pairs = read_json(os.path.join(path, "cooperative/data_info.json"))
    sensortype = 'camera'
    
    inf_path2info = build_path_to_info(
            "infrastructure-side",
            read_json(osp.join(path, "infrastructure-side/data_info.json")),
            sensortype,
        )
    veh_path2info = build_path_to_info(
            "vehicle-side",
            read_json(osp.join(path, "vehicle-side/data_info.json")),
            sensortype,
        )

    data = []
    inf_frames = {}
    veh_frames = {}

    for elem in tqdm(frame_pairs):
        if sensortype == "lidar":
            inf_frame = inf_path2info[elem["infrastructure_pointcloud_path"]]
            veh_frame = veh_path2info[elem["vehicle_pointcloud_path"]]
        elif sensortype == "camera":
            inf_frame = inf_path2info[elem["infrastructure_image_path"]]
            veh_frame = veh_path2info[elem["vehicle_image_path"]]
            
        inf_frame = InfFrame(path + "/infrastructure-side/", inf_frame)
        veh_frame = VehFrame(path + "/vehicle-side/", veh_frame)
        
        if not inf_frame["batch_id"] in inf_frames:
            inf_frames[inf_frame["batch_id"]] = [inf_frame]
        else:
            inf_frames[inf_frame["batch_id"]].append(inf_frame)
        if not veh_frame["batch_id"] in veh_frames:
            veh_frames[veh_frame["batch_id"]] = [veh_frame]
        else:
            veh_frames[veh_frame["batch_id"]].append(veh_frame)
        
        vic_frame = VICFrame(path, elem, veh_frame, inf_frame, 0)
                    
        label = Label_kitti(osp.join(path, elem["cooperative_label_path"]), None)
        
        trans_1 = vic_frame.transform("World", "Vehicle_lidar")
        label["veh_boxes_3d"] = trans_1(label["boxes_3d"])
        trans_2 = vic_frame.transform("World", "Infrastructure_lidar")
        label["inf_boxes_3d"] = trans_2(label["boxes_3d"])
        
        labels_path = veh_frame["label_" + label_type + "_std_path"]
        
        r_velo2cam, t_velo2cam = get_lidar2cam_2(
            osp.join(path, "vehicle-side", veh_frame["calib_lidar_to_camera_path"])
        )
        Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam))
        
        labels_new = list()
        for i in range(len(label['boxes_3d'])):
            label_new = dict()
            label_new['3d_dimensions'] = dict()
            label_new['3d_location'] = dict()
            
            gt_box = label['veh_boxes_3d'][i]
            label_new['type'] = label['types'][i]
            label_new['occluded_state'] = label['occluded_states'][i]
            label_new['truncated_state'] = label['truncated_states'][i]
            label_new['alpha'] = label['alphas'][i]
            label_new['2d_box'] = label['2d_boxes'][i]
                
                
            x, y, z, l, w, h, yaw_lidar = get_xyzlwhy(gt_box)
            z = z - h / 2
            bottom_center = [x, y, z]
            obj_size = [l, w, h]
            
            label_new['rotation'] = yaw_lidar
            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(
                obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
            )
            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)
            
            

            set_label(label_new, h, w, l, cam_x, cam_y, cam_z, alpha, yaw)
            
            labels_new.append(label_new)
            
        labels_path = labels_path.replace("virtuallidar", "lidar")
        write_path = os.path.join(target_root, labels_path)
        
        if os.path.exists(labels_path):
            print(labels_path)

        # from IPython import embed
        # embed(header='111')
        
        
        with open(write_path, "w") as f:
            json.dump(labels_new, f)
            
            
            
def gen_calib2kitti_coop(source_root, target_root, label_type="lidar"):
    write_path = os.path.join(target_root, "label", label_type)
    mkdir_p(write_path)
    
    path_calib = os.path.join(target_root, "training/calib")
    mkdir_p(path_calib)
    
    path = 'data/DAIR-V2X/cooperative-vehicle-infrastructure'
    frame_pairs = read_json(os.path.join(path, "cooperative/data_info.json"))
    sensortype = 'camera'
    
    inf_path2info = build_path_to_info(
            "infrastructure-side",
            read_json(osp.join(path, "infrastructure-side/data_info.json")),
            sensortype,
        )
    veh_path2info = build_path_to_info(
            "vehicle-side",
            read_json(osp.join(path, "vehicle-side/data_info.json")),
            sensortype,
        )

    data = []
    inf_frames = {}
    veh_frames = {}

    for elem in tqdm(frame_pairs):
        if sensortype == "lidar":
            inf_frame = inf_path2info[elem["infrastructure_pointcloud_path"]]
            veh_frame = veh_path2info[elem["vehicle_pointcloud_path"]]
        elif sensortype == "camera":
            inf_frame = inf_path2info[elem["infrastructure_image_path"]]
            veh_frame = veh_path2info[elem["vehicle_image_path"]]
            
        inf_frame = InfFrame(path + "/infrastructure-side/", inf_frame)
        veh_frame = VehFrame(path + "/vehicle-side/", veh_frame)
        
        if not inf_frame["batch_id"] in inf_frames:
                inf_frames[inf_frame["batch_id"]] = [inf_frame]
        else:
            inf_frames[inf_frame["batch_id"]].append(inf_frame)
        if not veh_frame["batch_id"] in veh_frames:
            veh_frames[veh_frame["batch_id"]] = [veh_frame]
        else:
            veh_frames[veh_frame["batch_id"]].append(veh_frame)
        
        vic_frame = VICFrame(path, elem, veh_frame, inf_frame, 0)
                    
        label = Label_kitti(osp.join(path, elem["cooperative_label_path"]), None)
        # from IPython import embed
        # embed(header='xxx')
        trans_1 = vic_frame.transform("World", "Vehicle_lidar")
        label["veh_boxes_3d"] = trans_1(label["boxes_3d"])
        trans_2 = vic_frame.transform("World", "Infrastructure_lidar")
        label["inf_boxes_3d"] = trans_2(label["boxes_3d"])
        
        # labels_path = veh_frame["label_" + label_type + "_std_path"]
        labels_path = inf_frame["label_" + label_type + "_std_path"]
        
        
        # fetch transformation matrixs
        # intrinstic 3*3
        veh_cam_D, veh_cam_K = get_cam_D_and_cam_K(osp.join(path, "vehicle-side", veh_frame["calib_camera_intrinsic_path"]))
        inf_cam_D, inf_cam_K = get_cam_D_and_cam_K(osp.join(path, "infrastructure-side", inf_frame["calib_camera_intrinsic_path"]))
        
        # from IPython import embed
        # embed()
        
        # lidar2cam 3*4
        r_veh_lidar2cam, t_veh_lidar2cam = get_lidar2cam_2(
            osp.join(path, "vehicle-side", veh_frame["calib_lidar_to_camera_path"])
        )

        r_inf_lidar2cam, t_inf_lidar2cam = get_lidar2cam_2(
            osp.join(path, "infrastructure-side", inf_frame["calib_virtuallidar_to_camera_path"])
        )


        # world2lidar 3*4
        veh_trans = vic_frame.transform("World", "Vehicle_lidar") 
        r_world2veh_lidar, t_world2veh_lidar = veh_trans.get_transformation()
        
        inf_trans = vic_frame.transform("World", "Infrastructure_lidar")
        r_world2inf_lidar, t_world2inf_lidar = inf_trans.get_transformation()


        # inf_trans2 = vic_frame.transform("Infrastructure_lidar", "World")
        # r_inf_lidar2world, t_inf_lidar2world = inf_trans2.get_transformation()


        # inf2veh_trans = vic_frame.transform("Infrastructure_lidar", "Vehicle_lidar")
        # r_inf_lidar2veh_lidar, t_inf_lidar2veh_lidar = inf2veh_trans.get_transformation()

        # r_new,t_new = inf2veh_trans.muilt_coord(r_inf_lidar2world,t_inf_lidar2world,r_world2veh_lidar,t_world2veh_lidar)

        # assert (r_inf_lidar2veh_lidar == r_new).all()
        # assert (t_inf_lidar2veh_lidar == t_new).all()
        

        txt_name = elem["vehicle_image_path"].split("/")[-1].replace(".jpg", ".txt")
        txt_path = os.path.join(path_calib, txt_name)

        
        P2, veh_Tr_velo_to_cam = convert_calib_v2x_to_kitti(veh_cam_D, veh_cam_K, t_veh_lidar2cam, r_veh_lidar2cam)
        P3, inf_Tr_velo_to_cam = convert_calib_v2x_to_kitti(inf_cam_D, inf_cam_K, t_inf_lidar2cam, r_inf_lidar2cam)
        _, world2veh_lidar = convert_calib_v2x_to_kitti(veh_cam_D, veh_cam_K, t_world2veh_lidar, r_world2veh_lidar)
        _, world2inf_lidar = convert_calib_v2x_to_kitti(inf_cam_D, inf_cam_K, t_world2inf_lidar, r_world2inf_lidar)

        # print('infrastructure_image_path:',elem["infrastructure_image_path"])
        # print('vehicle_image_path',elem["vehicle_image_path"])
        # print('P2:',P2)
        # print('P3:',P3)
        # print('veh_Tr_velo_to_cam:',veh_Tr_velo_to_cam)
        # print('inf_Tr_velo_to_cam:',inf_Tr_velo_to_cam)
        # print('world2veh_lidar',world2veh_lidar)
        # print('world2inf_lidar',world2inf_lidar)
        
        # print('\n')
        
        str_P2 = "P2: "
        str_P3 = "P3: "
        str_veh_Tr_velo_to_cam = "veh_Tr_velo_to_cam: "
        str_inf_Tr_velo_to_cam = "inf_Tr_velo_to_cam: "
        str_world2veh_lidar = "world2veh_lidar: "
        str_world2inf_lidar = "world2inf_lidar: "
        
        for ii in range(11):
            str_P2 = str_P2 + str(P2[ii]) + " "
            str_P3 = str_P3 + str(P3[ii]) + " "
            str_veh_Tr_velo_to_cam = str_veh_Tr_velo_to_cam + str(veh_Tr_velo_to_cam[ii]) + " "
            str_inf_Tr_velo_to_cam = str_inf_Tr_velo_to_cam + str(inf_Tr_velo_to_cam[ii]) + " "
            str_world2veh_lidar = str_world2veh_lidar + str(world2veh_lidar[ii]) + " "
            str_world2inf_lidar = str_world2inf_lidar + str(world2inf_lidar[ii]) + " "
        str_P2 = str_P2 + str(P2[11])
        str_P3 = str_P3 + str(P3[11])
        str_veh_Tr_velo_to_cam = str_veh_Tr_velo_to_cam + str(veh_Tr_velo_to_cam[11])
        str_inf_Tr_velo_to_cam = str_inf_Tr_velo_to_cam + str(inf_Tr_velo_to_cam[11])
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
                + str_world2veh_lidar
                + "\n"
                + str_world2inf_lidar
                + "\n"
                + str_Tr_imu_to_velo
            )
            fp.write(gt_line)