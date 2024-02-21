# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.core import bbox3d2result, build_prior_generator
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet.models.detectors import BaseDetector
from mmdet3d.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet3d.models.detectors.vicfuser_voxel.vicfuser_voxel_msca_c_ccm import DCN_Up_Conv_List, SS_NaiveCompressor, MultiScaleBlock


@DETECTORS.register_module()
class ImVoxelNet_Inf(BaseDetector):
    r"""`ImVoxelNet <https://arxiv.org/abs/2106.01178>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 neck_3d,
                 bbox_head,
                 n_voxels,
                 anchor_generator,
                 compress_ratio=1,
                 s_compress_ratio=0,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 neck_dcn=None):
        print('ImVoxelNet_Inf_0204).__init__')
        super().__init__(init_cfg=init_cfg)
        self.backbone_i = build_backbone(backbone)
        self.neck_i = build_neck(neck)
        self.neck_3d_i = build_neck(neck_3d)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.n_voxels = n_voxels
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.c_compress_ratio = compress_ratio
        self.s_compress_ratio = s_compress_ratio
        self.img_feat_channels = neck.out_channels

        self.dcn_up_conv_i = DCN_Up_Conv_List(neck_dcn, self.img_feat_channels)
        self.inf_compressor = SS_NaiveCompressor(self.img_feat_channels, self.c_compress_ratio, self.s_compress_ratio)
        self.ms_block_inf = MultiScaleBlock(self.img_feat_channels,self.img_feat_channels)

    def extract_feat(self, img, img_metas):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            torch.Tensor: of shape (N, C_out, N_x, N_y, N_z)
        """
        # from IPython import embed
        # embed(header='ffff')

        x_i = self.backbone_i(img)
        x_i = self.neck_i(x_i)[0]
        x_i = self.inf_compressor(x_i)
        x_i = self.ms_block_inf(x_i)
        x_i = self.dcn_up_conv_i(list(x_i))
        x_i_tensor = torch.stack(x_i).permute(1,0,2,3,4)
        x_i = torch.mean(x_i_tensor,dim=1)

        # x_i = self.backbone_i(img)
        # x_i = self.neck_i(x_i)
        # x_i = self.dcn_up_conv_i(list(x_i))
        # x_i_tensor = torch.stack(x_i).permute(1,0,2,3,4)
        # x_i = torch.mean(x_i_tensor,dim=1)

        points = self.anchor_generator.grid_anchors(
            [self.n_voxels[::-1]], device=img.device)[0][:, :3]
        volumes_i = []
        for feature, img_meta in zip(x_i, img_metas):
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
                proj_mat=points.new_tensor(img_meta['lidar2img']),
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
        x_i = self.neck_3d_i(x_i)
        return x_i

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
