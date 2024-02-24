import numpy as np

from ..v2x_utils import get_3d_8points
from .file_io import load_json
from ..v2x_utils.config import name2id


class Label(dict):
    def __init__(self, path, filt):
        raw_labels = load_json(path)
        boxes = []
        class_types = []
        for label in raw_labels:
            size = label["3d_dimensions"]
            if size["l"] == 0 or size["w"] == 0 or size["h"] == 0:
                continue
            if "world_8_points" in label:
                box = label["world_8_points"]
            else:
                pos = label["3d_location"]
                box = get_3d_8points(
                    [float(size["l"]), float(size["w"]), float(size["h"])],
                    float(label["rotation"]),
                    [float(pos["x"]), float(pos["y"]), float(pos["z"]) - float(size["h"]) / 2],
                ).tolist()
            # determine if box is in extended range
            if filt is None or filt(box):
                boxes.append(box)
                class_types.append(name2id[label["type"].lower()])
        boxes = np.array(boxes)
        class_types = np.array(class_types)
        # if len(class_types) == 1:
        #     boxes = boxes[np.newaxis, :]
        self.__setitem__("boxes_3d", boxes)
        self.__setitem__("labels_3d", class_types)
        self.__setitem__("scores_3d", np.ones_like(class_types))


class Label_kitti(dict):
    def __init__(self, path, filt):
        raw_labels = load_json(path)
        boxes = []
        types = []
        occluded_states = []
        truncated_states = []
        alphas = []
        class_types = []
        boxes_2d = []
        for label in raw_labels:
            size = label["3d_dimensions"]
            if size["l"] == 0 or size["w"] == 0 or size["h"] == 0:
                continue
            if "world_8_points" in label:
                box = label["world_8_points"]
            else:
                pos = label["3d_location"]
                box = get_3d_8points(
                    [float(size["l"]), float(size["w"]), float(size["h"])],
                    float(label["rotation"]),
                    [float(pos["x"]), float(pos["y"]), float(pos["z"]) - float(size["h"]) / 2],
                ).tolist()
            # determine if box is in extended range
            if filt is None or filt(box):
                types.append(label['type'])
                occluded_states.append(label['occluded_state'])
                truncated_states.append(label['truncated_state'])
                alphas.append(label['alpha'])
                boxes_2d.append(label['2d_box'])
                boxes.append(box)
                class_types.append(name2id[label["type"].lower()])
        boxes = np.array(boxes)
        class_types = np.array(class_types)
        # if len(class_types) == 1:
        #     boxes = boxes[np.newaxis, :]
        self.__setitem__("boxes_3d", boxes)
        self.__setitem__("labels_3d", class_types)
        self.__setitem__("scores_3d", np.ones_like(class_types))
        self.__setitem__("types", types)
        self.__setitem__("occluded_states", occluded_states)
        self.__setitem__("truncated_states", truncated_states)
        self.__setitem__("alphas", alphas)
        self.__setitem__("2d_boxes", boxes_2d)