#!/usr/bin/env bash

# WORK_DIR="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/work_dirs/1105_vicfuser_voxel_r50_960x540_12e_bs2x2"
# CONFIG_PATH="vicfuser_voxel_r50_960x540_12e_bs2_for_vis.py"
# PKL_PATH="eval_resutls.pkl"

# VICFuser
WORK_DIR="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/work_dirs/0129_vicfuser_voxel_msca_sc_c64s1_ccm_r50_960x540_12e_bs2x1"
CONFIG_PATH="vicfuser_voxel_msca_c_ccm_r50_960x540_12e_bs2_for_vis.py"
PKL_PATH="20230207_213432_eval_results.pkl"

# # LF_V
# WORK_DIR="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/work_dirs/LF/1020_vic_coop_v_imvoxelnet"
# CONFIG_PATH="vic_coop_v_imvoxelnet.py"
# PKL_PATH="20230218_154710_eval_results.pkl"

# # LF_I
# WORK_DIR="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/work_dirs/LF/1219_vic_coop_i_imvoxelnet_delta_filter"
# CONFIG_PATH="vic_coop_i_imvoxelnet.py"
# PKL_PATH="20230220_114111_eval_results.pkl"

# # ImVoxelNet_VIC
# WORK_DIR="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/work_dirs/0201_imvoxelnet_vic_r50_960x540_12e_bs2x1"
# CONFIG_PATH="imvoxelnet_vic_r50_960x540_12e_bs2_for_vis.py"
# PKL_PATH="20230207_214829_eval_results.pkl"

# # DAIR_Coop_V
# WORK_DIR="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/archive/1020_dair_coop_v_imvoxelnet"
# CONFIG_PATH="dair_coop_v_imvoxelnet.py"
# PKL_PATH="20230218_161327_eval_results.pkl"

# # DAIR_Coop_I
# WORK_DIR="/home/wangz/wangzhe21/HBEVFuser/mmdet3d_v1_0_0rc4/mmdetection3d/archive/1020_dair_coop_i_imvoxelnet"
# CONFIG_PATH="dair_coop_i_imvoxelnet.py"
# PKL_PATH="20230218_163059_eval_results.pkl"


python $(dirname "$0")/misc/visualize_results_dair.py \
  $WORK_DIR/$CONFIG_PATH \
  --result $WORK_DIR/$PKL_PATH \
  --show-dir 0129_vimi_label_pred


