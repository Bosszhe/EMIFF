# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.core import bbox3d2result, build_prior_generator
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet.models.detectors import BaseDetector
from mmdet3d.models.builder import DETECTORS, build_backbone, build_head, build_neck,build_voxel_encoder
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmdet.models.utils import build_transformer



@DETECTORS.register_module()
class VICFuser_DEFORM(BaseDetector):
    r"""`ImVoxelNet <https://arxiv.org/abs/2106.01178>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 transformer,
                 bbox_head,
                #  n_voxels,
                #  anchor_generator,
                 bev_h= 150,
                 bev_w= 150,
                #  num_query=900,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        print('VICFuser_DEFORM.__init__')
        super().__init__(init_cfg=init_cfg)


        # from IPython import embed
        # embed(header='xxx')

        self.backbone_v = build_backbone(backbone)
        self.neck_v = build_neck(neck)
        self.backbone_i = build_backbone(backbone)
        self.neck_i = build_neck(neck)
        
        # self.neck_3d = build_neck(neck_3d)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        # self.n_voxels = n_voxels
        # self.anchor_generator = build_prior_generator(anchor_generator)
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

        # from IPython import embed
        # embed(header='VICFuser_DEFORM.__init__')

        self.pc_range = transformer.encoder.pc_range


        self.real_h = self.pc_range[3] - self.pc_range[0]
        self.real_w = self.pc_range[4] - self.pc_range[1]
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = transformer.embed_dims
        # self.num_query = num_query


        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        # self.query_embedding = nn.Embedding(self.num_query,self.embed_dims * 2)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)


    # @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img, img_metas):
        """Extract features from images."""

        img_v = img[:,0,...]
        img_i = img[:,1,...]
        
        
        x_vs = self.backbone_v(img_v)
        x_vs = self.neck_v(x_vs)
        
        x_is = self.backbone_i(img_i)
        x_is = self.neck_i(x_is)

        # from IPython import embed
        # embed(header='extract_img_feat')
        # N = 2
        img_feats = list()
        for i in range(len(x_vs)):
            # B, C, H, W = x_vs[i].size()
            # img_feat = torch.stack((x_vs[i],x_is[i]),dim=1).permute(1,0,2,3,4) #B,2,C,H,W
            # img_feat = img_feat.contiguous().view(B*N,C,H,W)
            
            
            img_feat = torch.stack((x_vs[i],x_is[i]),dim=1)
            img_feats.append(img_feat)

        return img_feats


    def extract_feat(self, img, img_metas):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        Args:
            img (torch.Tensor): Input images of shape (N, Num_Cam, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            torch.Tensor: of shape (N, C_out, N_x, N_y, N_z)
        """

        batch_size = img.shape[0]
        img_feats = self.extract_img_feat(img, img_metas)

        # from IPython import embed
        # embed(header='VICFuser_DEFORM.extract_feat')
        mlvl_feats = img_feats

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        bev_features = self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=None,
            )

        bev_features = bev_features.view(batch_size,self.bev_h,self.bev_w,self.embed_dims).permute(0,3,2,1)
        # from IPython import embed
        # embed(header='it works.extract_feat')


        # x = self.neck_3d(x)   
        # Anchor3DHead axis order is (y, x). 
        # bev_feature [B,C,bev_w,bev_h]
        return [bev_features]

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
        # embed(header='VICFuser_BEV.forward_train')
        
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
        # embed(header='VICFuser_BEV.forward_test')
        
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
