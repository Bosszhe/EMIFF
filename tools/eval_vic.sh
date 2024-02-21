#!/usr/bin/env bash

# VEHICLE_MODEL_PATH="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/work_dirs/0227_vicfuser_voxel_r50_960x540_12e_bs2x2"
# VEHICLE_CONFIG_NAME="vicfuser_voxel_r50_960x540_12e_bs2.py"
# VEHICLE_MODEL_NAME="epoch_12.pth"

# VEHICLE_MODEL_PATH="/home/wangz/wangzhe21/VIMI/work_dirs/0129_VIMI"
# VEHICLE_CONFIG_NAME="vimi_r50_960x540_12e_bs2.py"
# VEHICLE_MODEL_NAME="best_car_3d_0.5_epoch_12.pth"


# VEHICLE_MODEL_PATH="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/work_dirs/0201_imvoxelnet_vic_r50_960x540_12e_bs2x1"
# VEHICLE_CONFIG_NAME="imvoxelnet_vic_r50_960x540_12e_bs2.py"
# VEHICLE_MODEL_NAME="epoch_12.pth"

# VEHICLE_MODEL_PATH="/home/wangz/wangzhe21/VIMI/work_dirs/0515_VIMI_VQVAE_960x540_12e_bs2x1"
# VEHICLE_CONFIG_NAME="vimi_vqvae_960x540_12e_bs2.py"
# VEHICLE_MODEL_NAME="best_car_3d_0.5_epoch_10.pth"

VEHICLE_MODEL_PATH="/home/wangz/wangzhe21/VIMI/work_dirs/0710_VIMI_VQVAE_Veh_960x540_12e_bs2x1"
VEHICLE_CONFIG_NAME="vimi_vqvae_veh_960x540_12e_bs2.py"
VEHICLE_MODEL_NAME="best_car_3d_0.5_epoch_10.pth"

# python $(dirname "$0")/test_vic.py \
#   $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
#   $VEHICLE_MODEL_PATH/$VEHICLE_MODEL_NAME \
#   --eval mAP 

  python $(dirname "$0")/test_vic.py \
  $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
  $VEHICLE_MODEL_PATH/$VEHICLE_MODEL_NAME \
  --eval mAP 
  # --show
  # --skip




# #!/usr/bin/env bash

# VEHICLE_MODEL_PATH="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/work_dirs/1104_vic_re3_imvoxelnet_r50_960x544_12e_bs2x4"
# VEHICLE_CONFIG_NAME="vic_re3_imvoxelnet_r50_960x544_12e_bs2x4.py"
# VEHICLE_MODEL_NAME="epoch_12.pth"

# VEHICLE_MODEL_PATH="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/work_dirs/vicfuser_archive/1020_vic_coop_v_imvoxelnet"
# VEHICLE_CONFIG_NAME="vic_coop_v_imvoxelnet.py"
# VEHICLE_MODEL_NAME="epoch_11.pth"


# VEHICLE_MODEL_PATH="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/archive/1020_dair_coop_v_imvoxelnet"
# VEHICLE_CONFIG_NAME="dair_coop_v_imvoxelnet.py"
# VEHICLE_MODEL_NAME="epoch_12.pth"


# VEHICLE_MODEL_PATH="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/work_dirs/LF/1219_vic_coop_i_imvoxelnet_delta_filter"
# VEHICLE_CONFIG_NAME="vic_coop_i_imvoxelnet.py"
# VEHICLE_MODEL_NAME="epoch_12.pth"

# python $(dirname "$0")/test.py \
#   $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
#   $VEHICLE_MODEL_PATH/$VEHICLE_MODEL_NAME \
#   --eval mAP 
#   # --out $VEHICLE_MODEL_PATH
#   # --show
#   # --skip


