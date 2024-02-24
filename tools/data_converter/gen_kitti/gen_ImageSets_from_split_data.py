import os
from .utils import read_json, mkdir_p, write_txt
from .gen_calib2kitti_coop import load_json


def gen_ImageSet_from_split_data(ImageSets_path, split_data_path, sensor_view="vehicle"):
    split_data = read_json(split_data_path)
    test_file = ""
    train_file = ""
    val_file = ""
    
    
    # from IPython import embed
    # embed(header='imagesets')
    
    if "vehicle_split" in split_data.keys():
        sensor_view = sensor_view + "_split"
        split_data = split_data[sensor_view]
    
    # split_data = split_data['cooperative_split']
    for i in range(len(split_data["train"])):
        name = split_data["train"][i]
        train_file = train_file + name + "\n"

    for i in range(len(split_data["val"])):
        name = split_data["val"][i]
        val_file = val_file + name + "\n"
        
        
    # The test part of the dataset has not been released
    # for i in range(len(split_data["test"])):
    #     name = split_data["test"][i]
    #     test_file = test_file + name + "\n"

    trainval_file = train_file + val_file

    mkdir_p(ImageSets_path)
    write_txt(os.path.join(ImageSets_path, "test.txt"), test_file)
    write_txt(os.path.join(ImageSets_path, "trainval.txt"), trainval_file)
    write_txt(os.path.join(ImageSets_path, "train.txt"), train_file)
    write_txt(os.path.join(ImageSets_path, "val.txt"), val_file)


def gen_ImageSet_from_coop_split_data(ImageSets_path, split_data_path, sensor_view="vehicle"):
    split_datas = read_json(split_data_path)
    veh_test_file = ""
    veh_train_file = ""
    veh_val_file = ""
    
    inf_test_file = ""
    inf_train_file = ""
    inf_val_file = ""
    
    # from IPython import embed
    # embed(header='imagesets')
    
    frame_pairs = load_json('data/DAIR-V2X/cooperative-vehicle-infrastructure/cooperative/data_info.json')
    
    split_data = split_datas["cooperative_split"]['train']
    inf_train_split = []
    veh_train_split = []
    for frame_pair in frame_pairs:
        veh_frame_idx = frame_pair["vehicle_image_path"].split("/")[-1].replace(".jpg", "")
        if veh_frame_idx in split_data:
            inf_frame_idx = frame_pair["infrastructure_image_path"].split("/")[-1].replace(".jpg", "")
            inf_train_split.append(inf_frame_idx)
            veh_train_split.append(veh_frame_idx)
            
    split_data = split_datas["cooperative_split"]['val']
    inf_val_split = []
    veh_val_split = []
    for frame_pair in frame_pairs:
        veh_frame_idx = frame_pair["vehicle_image_path"].split("/")[-1].replace(".jpg", "")
        if veh_frame_idx in split_data:
            inf_frame_idx = frame_pair["infrastructure_image_path"].split("/")[-1].replace(".jpg", "")
            inf_val_split.append(inf_frame_idx)
            veh_val_split.append(veh_frame_idx)
            
    split_data = split_datas["cooperative_split"]['test']
    inf_test_split = []
    veh_test_split = []
    for frame_pair in frame_pairs:
        veh_frame_idx = frame_pair["vehicle_image_path"].split("/")[-1].replace(".jpg", "")
        if veh_frame_idx in split_data:
            inf_frame_idx = frame_pair["infrastructure_image_path"].split("/")[-1].replace(".jpg", "")
            inf_test_split.append(inf_frame_idx)
            veh_test_split.append(veh_frame_idx)
            
    # veh_train_split.sort()       
    # inf_train_split.sort()
    # veh_val_split.sort()       
    # inf_val_split.sort()
    # veh_test_split.sort()       
    # inf_test_split.sort()
    
            
    # from IPython import embed
    # embed(header='imagesets')
    
    for i in range(len(veh_train_split)):
        name1 = veh_train_split[i]
        name2= inf_train_split[i]
        veh_train_file = veh_train_file + name1 + "\n"
        inf_train_file = inf_train_file + name2 + "\n"
        
    for i in range(len(veh_val_split)):
        name1 = veh_val_split[i]
        name2 = inf_val_split[i]
        veh_val_file = veh_val_file + name1 + "\n"
        inf_val_file = inf_val_file + name2 + "\n"
        
    for i in range(len(veh_test_split)):
        name1 = veh_test_split[i]
        name2 = inf_test_split[i]
        veh_test_file = veh_test_file + name1 + "\n"
        inf_test_file = inf_test_file + name2 + "\n"
    

    veh_trainval_file = veh_train_file + veh_val_file
    inf_trainval_file = inf_train_file + inf_val_file
    
    ImageSets_veh_path = os.path.join(ImageSets_path, "veh")
    ImageSets_inf_path = os.path.join(ImageSets_path, "inf")
    mkdir_p(ImageSets_inf_path)
    mkdir_p(ImageSets_veh_path)
    write_txt(os.path.join(ImageSets_veh_path, "test.txt"), veh_test_file)
    write_txt(os.path.join(ImageSets_veh_path, "trainval.txt"), veh_trainval_file)
    write_txt(os.path.join(ImageSets_veh_path, "train.txt"), veh_train_file)
    write_txt(os.path.join(ImageSets_veh_path, "val.txt"), veh_val_file)
    write_txt(os.path.join(ImageSets_inf_path, "test.txt"), inf_test_file)
    write_txt(os.path.join(ImageSets_inf_path, "trainval.txt"), inf_trainval_file)
    write_txt(os.path.join(ImageSets_inf_path, "train.txt"), inf_train_file)
    write_txt(os.path.join(ImageSets_inf_path, "val.txt"), inf_val_file)


def seq_gen_ImageSet_from_coop_split_data(ImageSets_path, split_data_path, sensor_view="vehicle"):
    split_datas = read_json(split_data_path)
    veh_test_file = ""
    veh_train_file = ""
    veh_val_file = ""
    
    inf_test_file = ""
    inf_train_file = ""
    inf_val_file = ""
    
    # from IPython import embed
    # embed(header='imagesets')
    
    frame_pairs = load_json('data/DAIR_V2X_Seq/SPD/cooperative-vehicle-infrastructure/cooperative/data_info.json')
    
    split_data = split_datas["cooperative_split"]['train']
    inf_train_split = []
    veh_train_split = []
    for frame_pair in frame_pairs:
        veh_frame_idx = frame_pair["vehicle_frame"]
        if veh_frame_idx in split_data:
            inf_frame_idx = frame_pair["infrastructure_frame"]
            inf_train_split.append(inf_frame_idx)
            veh_train_split.append(veh_frame_idx)
            
    split_data = split_datas["cooperative_split"]['val']
    inf_val_split = []
    veh_val_split = []
    for frame_pair in frame_pairs:
        veh_frame_idx = frame_pair["vehicle_frame"]
        if veh_frame_idx in split_data:
            inf_frame_idx = frame_pair["infrastructure_frame"]
            inf_val_split.append(inf_frame_idx)
            veh_val_split.append(veh_frame_idx)
            
    split_data = split_datas["cooperative_split"]['test']
    inf_test_split = []
    veh_test_split = []
    for frame_pair in frame_pairs:
        veh_frame_idx = frame_pair["vehicle_frame"]
        if veh_frame_idx in split_data:
            inf_frame_idx = frame_pair["infrastructure_frame"]
            inf_test_split.append(inf_frame_idx)
            veh_test_split.append(veh_frame_idx)
            

    
    for i in range(len(veh_train_split)):
        name1 = veh_train_split[i]
        name2= inf_train_split[i]
        veh_train_file = veh_train_file + name1 + "\n"
        inf_train_file = inf_train_file + name2 + "\n"
        
    for i in range(len(veh_val_split)):
        name1 = veh_val_split[i]
        name2 = inf_val_split[i]
        veh_val_file = veh_val_file + name1 + "\n"
        inf_val_file = inf_val_file + name2 + "\n"
        
    for i in range(len(veh_test_split)):
        name1 = veh_test_split[i]
        name2 = inf_test_split[i]
        veh_test_file = veh_test_file + name1 + "\n"
        inf_test_file = inf_test_file + name2 + "\n"
    

    veh_trainval_file = veh_train_file + veh_val_file
    inf_trainval_file = inf_train_file + inf_val_file
    
    ImageSets_veh_path = os.path.join(ImageSets_path, "veh")
    ImageSets_inf_path = os.path.join(ImageSets_path, "inf")
    mkdir_p(ImageSets_inf_path)
    mkdir_p(ImageSets_veh_path)
    write_txt(os.path.join(ImageSets_veh_path, "test.txt"), veh_test_file)
    write_txt(os.path.join(ImageSets_veh_path, "trainval.txt"), veh_trainval_file)
    write_txt(os.path.join(ImageSets_veh_path, "train.txt"), veh_train_file)
    write_txt(os.path.join(ImageSets_veh_path, "val.txt"), veh_val_file)
    write_txt(os.path.join(ImageSets_inf_path, "test.txt"), inf_test_file)
    write_txt(os.path.join(ImageSets_inf_path, "trainval.txt"), inf_trainval_file)
    write_txt(os.path.join(ImageSets_inf_path, "train.txt"), inf_train_file)
    write_txt(os.path.join(ImageSets_inf_path, "val.txt"), inf_val_file)