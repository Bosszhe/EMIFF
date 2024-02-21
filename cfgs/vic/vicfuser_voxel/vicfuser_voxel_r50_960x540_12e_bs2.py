dataset_type = 'DAIR_VIC_Dataset'
data_root = 'data/1026_vic/'
class_names = ['Car']
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [0, -39.68, -3, 92.16, 39.68, 1]
extended_range = [0, -40.0, -3, 100, 40.0, 1]
voxel_size = [0.32, 0.32, 0.33]
length = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
width = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
height = int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
output_shape = [width, length, height]
img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (960, 540)
img_resize_scale = [(912, 513), (1008, 567)]

_dim_ = 64
model = dict(
    type='VICFuser_Voxel',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        num_outs=4),
    neck_3d=dict(type='OutdoorImVoxelNeck', in_channels=_dim_, out_channels=256),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -39.68, -1.78, 92.16, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    n_voxels=output_shape,
    anchor_generator=dict(
        type='AlignedAnchor3DRangeGenerator',
        ranges=[[0, -39.68, -3.08, 92.16, 39.68, 0.76]],
        rotations=[0.0]),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=2,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='Resize',
                img_scale=img_resize_scale,
                keep_ratio=True,
                multiscale_mode='range'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)]),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline_Test',
        n_images=2,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=img_scale, keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)]),
    dict(type='ObjectRangeFilter', point_cloud_range=extended_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img','gt_bboxes_3d', 'gt_labels_3d'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'dair_coop1214_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'dair_coop1214_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'dair_coop1214_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True))

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=35.0, norm_type=2))
lr_config = dict(policy='step', step=[8, 11])
total_epochs = 12

checkpoint_config = dict(interval=1, max_keep_ckpts=1)

run_name = '0227_vicfuser_voxel_r50_960x540_12e_bs2x2'
wandb_init_dict = dict(
    type='WandbLoggerHook',
    init_kwargs = dict(
        project='VICFuser',
        name=run_name))
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook'),
           wandb_init_dict
           ])
evaluation = dict(interval=1, start=1, save_best='car_3d_0.5', rule='greater')
dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/' + run_name
gpu_ids = range(0, 1)
