import json
import yaml
import pickle
import numpy as np
from pypcd import pypcd
import mmcv


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.load(f)
    return data


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(item, path):
    with open(path, "wb") as f:
        pickle.dump(item, f)


def read_pcd(pcd_path):
    pcd = pypcd.PointCloud.from_path(pcd_path)
    time = None
    pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
    pcd_np_points[:, 0] = np.transpose(pcd.pc_data["x"])
    pcd_np_points[:, 1] = np.transpose(pcd.pc_data["y"])
    pcd_np_points[:, 2] = np.transpose(pcd.pc_data["z"])
    pcd_np_points[:, 3] = np.transpose(pcd.pc_data["intensity"]) / 256.0
    del_index = np.where(np.isnan(pcd_np_points))[0]
    pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)
    return pcd_np_points, time


def read_jpg(jpg_path):
    image = mmcv.imread(jpg_path)
    return image


def get_cam_calib_intrinsic(calib_path):
    my_json = load_json(calib_path)
    cam_K = my_json["cam_K"]
    calib = np.zeros([3, 4])
    calib[:3, :3] = np.array(cam_K).reshape([3, 3], order="C")

    return calib

def get_lidar2cam(calib_path):
    calib = load_json(calib_path)

    if "Tr_velo_to_cam" in calib.keys():
        velo2cam = np.array(calib["Tr_velo_to_cam"]).reshape(3, 4)
        r_velo2cam = velo2cam[:, :3]
        t_velo2cam = velo2cam[:, 3].reshape(3, 1)
    else:
        r_velo2cam = np.array(calib["rotation"])
        t_velo2cam = np.array(calib["translation"])
    return r_velo2cam, t_velo2cam