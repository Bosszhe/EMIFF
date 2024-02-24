# DAIR-V2X
Download DAIR-V2X-C full dataset data following [OpenDAIRV2X](https://github.com/AIR-THU/DAIR-V2X/tree/main#dataset) or click [here](https://thudair.baai.ac.cn/task-cooptest) and organize as follows:

```

# For DAIR-V2X-C Dataset located at ${DAIR-V2X-C_DATASET_ROOT}
├── cooperative-vehicle-infrastructure      # DAIR-V2X-C
    ├── infrastructure-side             # DAIR-V2X-C-I
        ├── image		    
            ├── {id}.jpg
        ├── velodyne                
            ├── {id}.pcd           
        ├── calib                 
            ├── camera_intrinsic            
                ├── {id}.json     
            ├── virtuallidar_to_world   
                ├── {id}.json      
            ├── virtuallidar_to_camera  
                ├── {id}.json      
        ├── label	
            ├── camera                  # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in image based on image frame time
                ├── {id}.json
            ├── virtuallidar            # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
                ├── {id}.json
        ├── data_info.json              # Relevant index information of Infrastructure data
    ├── vehicle-side                    # DAIR-V2X-C-V
        ├── image		    
            ├── {id}.jpg
        ├── velodyne             
            ├── {id}.pcd           
        ├── calib                 
            ├── camera_intrinsic   
                ├── {id}.json
            ├── lidar_to_camera   
                ├── {id}.json
            ├── lidar_to_novatel  
                ├── {id}.json
            ├── novatel_to_world   
                ├── {id}.json
        ├── label	
            ├── camera                  # Labeled data in Vehicle LiDAR Coordinate System fitting objects in image based on image frame time
                ├── {id}.json
            ├── lidar                   # Labeled data in Vehicle LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
                ├── {id}.json
        ├── data_info.json              # Relevant index information of the Vehicle data
    ├── cooperative                     # Coopetative Files
        ├── label_world                 # Vehicle-Infrastructure Cooperative (VIC) Annotation files
            ├── {id}.json           
        ├── data_info.json              # Relevant index information combined the Infrastructure data and the Vehicle data
```

### Transform DAIR-V2X-C into KITTI format

Run the following command to convert DAIR-V2X-C into KITTI format
```python
python tools/dataset_converter/dair_vic2kitti_2.py
```


The folder `target-root` structure will be organized as follows after our processing.

```
EMIFF
├── mmdet3d
├── tools
├── configs
├── cfgs
├── data
│   ├── dair_vic_kitti_format
│   │   ├── ImageSets
│   │   │   ├── inf # Infrastructure ID 
│   │   │   ├── veh # Vehicle ID (Using for train/validation split)
│   │   ├── testing
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2 # Vehicle Image
│   │   │   ├── image_3 # Infrastructure Image
│   │   │   ├── label_2
│   │   │   ├── velodyne # Optional
```

Then generate pkl info files by running (the same as [Dataset Preparation](https://github.com/open-mmlab/mmdetection3d/blob/v0.17.3/docs/data_preparation.md))

```
python tools/create_data_dair.py kitti --root-path ./data/dair_vic_kitti_format --out-dir ./data/dair_vic_kitti_format --extra-tag dair_vic_kitti_format
```

Using the above code will generate `dair_vic_kitti_format_infos_{train,val}.pkl`.