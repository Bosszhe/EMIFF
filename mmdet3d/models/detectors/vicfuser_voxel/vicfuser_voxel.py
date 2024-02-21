# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.core import bbox3d2result, build_prior_generator
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet.models.detectors import BaseDetector
from mmdet3d.models.builder import DETECTORS, build_backbone, build_head, build_neck
from torch import nn
from mmcv.cnn import ConvModule


@DETECTORS.register_module()
class VICFuser_Voxel(BaseDetector):
    r"""`ImVoxelNet <https://arxiv.org/abs/2106.01178>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 neck_3d,
                 bbox_head,
                 n_voxels,
                 anchor_generator,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        print('VICFuser_Voxel_1226.__init__')
        super().__init__(init_cfg=init_cfg)
        self.backbone_v = build_backbone(backbone)
        self.neck_v = build_neck(neck)
        self.backbone_i = build_backbone(backbone)
        self.neck_i = build_neck(neck)
        
        self.neck_3d = build_neck(neck_3d)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.n_voxels = n_voxels
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # self.conv_o = nn.Sequential(ConvModule(
        #         in_channels=neck.out_channels * 2,
        #         out_channels=neck.out_channels,
        #         kernel_size=3,
        #         stride=(1, 1, 1),
        #         padding=1,
        #         conv_cfg=dict(type='Conv3d'),
        #         norm_cfg=dict(type='BN3d'),
        #         act_cfg=dict(type='ReLU', inplace=True)))

    def extract_feat(self, img, img_metas):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        Args:
            img (torch.Tensor): Input images of shape (N, Num_Cam, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            torch.Tensor: of shape (N, C_out, N_x, N_y, N_z)
        """
        
        # from IPython import embed
        # embed(header='VICFuser_BEV.extract_feat')

        batch_size = img.shape[0]
        img_v = img[:,0,...]
        img_i = img[:,1,...]
        
        
        x_v = self.backbone_v(img_v)
        x_v = self.neck_v(x_v)[0]
        
        x_i = self.backbone_i(img_i)
        x_i = self.neck_i(x_i)[0]
        
 
        points = self.anchor_generator.grid_anchors(
            [self.n_voxels[::-1]], device=img.device)[0][:, :3]

        volumes_v = []
        for feature, img_meta in zip(x_v, img_metas):

            proj_mat_ex0 = points.new_tensor(img_meta['lidar2img']['extrinsic'][0])
            proj_mat_in0 = points.new_tensor(img_meta['lidar2img']['intrinsic'][0])
            proj_mats0 = proj_mat_in0 @ proj_mat_ex0

            img_scale_factor = (
                points.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                points.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            volume_v = point_sample(
                img_meta,
                img_features=feature[None, ...],
                points=points,
                proj_mat=proj_mats0,
                coord_type='LIDAR',
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img.shape[-2:],
                img_shape=img_meta['img_shape'][:2],
                aligned=False)          
            volumes_v.append(
                volume_v.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0))
        x_v = torch.stack(volumes_v)

        volumes_i = []
        for feature, img_meta in zip(x_i, img_metas):
            proj_mat_ex1 = points.new_tensor(img_meta['lidar2img']['extrinsic'][1])
            proj_mat_in1 = points.new_tensor(img_meta['lidar2img']['intrinsic'][1])
            proj_mats1 = proj_mat_in1 @ proj_mat_ex1

            img_scale_factor = (
                points.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                points.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            volume_i = point_sample(
                img_meta,
                img_features=feature[None, ...],
                points=points,
                proj_mat=proj_mats1,
                coord_type='LIDAR',
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img.shape[-2:],
                img_shape=img_meta['img_shape'][:2],
                aligned=False)         
            volumes_i.append(
                volume_i.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0))
            
        x_i = torch.stack(volumes_i)

        assert x_v.shape[0] == batch_size, 'x_v shape[0] is not equal bs'
        assert x_i.shape[0] == batch_size, 'x_i shape[0] is not equal bs'
        assert x_v.shape == x_i.shape

        # from IPython import embed
        # embed(header='check bev_embedding!!!')
        # x_concat = torch.cat((x_v,x_i),dim=1)
        # x = self.conv_o(x_concat)
        
        # x.shape: [B,C,bev_h,bev_w,z] 
        x_stack = torch.stack((x_v,x_i),dim=0)
        x = torch.mean(x_stack,dim=0)
        
        x = self.neck_3d(x)   
        # x[0].shape: [B,C,bev_w,bev_h]
        return x

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d,
                      **kwargs):
        """Forward of training.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.

        Returns:
            dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # from IPython import embed
        # embed(header='VICFuser_Voxel.forward_train')
        
        x = self.extract_feat(img, img_metas)
        x = self.bbox_head(x)
        losses = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Forward of testing.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        
        # from IPython import embed
        # embed(header='VICFuser_Voxel.forward_test')
        
        # not supporting aug_test for now
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):
        """Test without augmentations.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        
        # from IPython import embed
        # embed(header='VICFuser_BEV.simple_test')
        x = self.extract_feat(img, img_metas)
        x = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*x, img_metas)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            imgs (list[torch.Tensor]): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
