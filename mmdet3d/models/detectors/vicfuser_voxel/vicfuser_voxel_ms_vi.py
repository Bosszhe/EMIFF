# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.core import bbox3d2result, build_prior_generator
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet.models.detectors import BaseDetector
from mmdet3d.models.builder import DETECTORS, build_backbone, build_head, build_neck
from torch import nn
from mmcv.cnn import ConvModule
from torch.nn import functional as F
from mmcv.runner import force_fp32, auto_fp16


class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# class Weights_MLP(nn.Module):

#     def __init__(self, in_channels,num_levels):
#         super(Weights_MLP, self).__init__()

#         self.w_mlp = nn.Sequential(
#             nn.Linear(in_channels * num_levels, num_levels),
#             # nn.BatchNorm1d(num_levels, eps=1e-3, momentum=0.01),
#         )

#     def forward(self, x):
#         B,N_levels,C,_H,W = x.shape
#         input = torch.mean(x,dim=(-2,-1)).view(B,N_levels * C)
#         weights = self.w_mlp(input)
#         weights = F.softmax(weights,dim=-1)
#         return weights


class DCN_Up_Conv_List(nn.Module):

    def __init__(self, neck_dcn, channels):
        super(DCN_Up_Conv_List, self).__init__()


        self.upconv0 = nn.Sequential(
            double_conv(channels,channels),
        )

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
        )

        self.dcn0 = build_neck(neck_dcn)
        self.dcn1 = build_neck(neck_dcn)
        self.dcn2 = build_neck(neck_dcn)
        self.dcn3 = build_neck(neck_dcn)

    def forward(self, x):
        assert x.__len__() == 4
        x0 = self.dcn0(x[0])
        x0 = self.upconv0(x0)

        x1 = self.dcn1(x[1])
        x1 = self.upconv1(x1)

        x2 = self.dcn2(x[2])
        x2 = self.upconv2(x2)

        x3 = self.dcn3(x[3])
        x3 = self.upconv3(x3)


        return [x0,x1,x2,x3]


@DETECTORS.register_module()
class VICFuser_Voxel_MS_VI(BaseDetector):
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
                 init_cfg=None,
                 neck_dcn=None):
        print('VICFuser_Voxel_MS_VI_1215.__init__')
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

        self.dcn_up_conv_v = DCN_Up_Conv_List(neck_dcn, neck.out_channels)
        self.dcn_up_conv_i = DCN_Up_Conv_List(neck_dcn, neck.out_channels)
        # self.weights_mlp_v = Weights_MLP(neck.out_channels, neck.num_outs)
        # self.weights_mlp_i = Weights_MLP(neck.out_channels, neck.num_outs)

        # self.dcn_v = build_neck(neck_dcn)
        # self.dcn_i = build_neck(neck_dcn)

        # self.conv_o = nn.Sequential(ConvModule(
        #         in_channels=neck.out_channels * 2,
        #         out_channels=neck.out_channels,
        #         kernel_size=3,
        #         stride=(1, 1, 1),
        #         padding=1,
        #         conv_cfg=dict(type='Conv3d'),
        #         norm_cfg=dict(type='BN3d'),
        #         act_cfg=dict(type='ReLU', inplace=True)))

    # @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img, img_metas):
        """Extract features from images."""
        bs = img.shape[0]
        img_v = img[:,0,...]
        img_i = img[:,1,...]
        
        
        x_v = self.backbone_v(img_v)
        x_v = self.neck_v(x_v)
        
        x_i = self.backbone_i(img_i)
        x_i = self.neck_i(x_i)
        

        x_v = self.dcn_up_conv_v(list(x_v))
        x_i = self.dcn_up_conv_i(list(x_i))

        # x_v (List,len=4) [B,C,H,W]
        x_v_tensor = torch.stack(x_v).permute(1,0,2,3,4)
        x_i_tensor = torch.stack(x_i).permute(1,0,2,3,4)

        # Mean 1
        x_v_out_mean1 = torch.mean(x_v_tensor,dim=1)
        x_i_out_mean1 = torch.mean(x_i_tensor,dim=1)
        
        return tuple((x_v_out_mean1, x_i_out_mean1))


    # @auto_fp16(apply_to=('img'), out_fp32=True)
    # def extract_img_feat(self, img, img_metas):
    #     """Extract features from images."""
    #     bs = img.shape[0]
    #     img_v = img[:,0,...]
    #     img_i = img[:,1,...]
        
        
    #     x_v = self.backbone_v(img_v)
    #     x_v = self.neck_v(x_v)
        
    #     x_i = self.backbone_i(img_i)
    #     x_i = self.neck_i(x_i)
        

    #     x_v = self.dcn_up_conv_v(list(x_v))
    #     x_i = self.dcn_up_conv_i(list(x_i))

    #     # x_v (List,len=4) [B,C,H,W]
    #     x_v_tensor = torch.stack(x_v).permute(1,0,2,3,4)
    #     x_i_tensor = torch.stack(x_i).permute(1,0,2,3,4)

    #     # from IPython import embed
    #     # embed(header='extract_img_feat')

    #     # MLP
    #     # x_i_tensor.shape [B,N_levels,C,H,W]
    #     # weights_i.shape [B,N_levels]
    #     weights_v = self.weights_mlp_v(x_v_tensor)
    #     weights_i = self.weights_mlp_i(x_i_tensor)
    #     assert x_v_tensor.shape[:2] == weights_v.shape
    #     assert x_i_tensor.shape[:2] == weights_i.shape
    #     x_v_out = (weights_v * x_v_tensor.permute(2,3,4,0,1)).permute(3,4,0,1,2).sum(dim=1)
    #     x_i_out = (weights_i * x_i_tensor.permute(2,3,4,0,1)).permute(3,4,0,1,2).sum(dim=1)

    #     # Mean 1
    #     x_v_out_mean1 = torch.mean(x_v_tensor,dim=1)
    #     x_i_out_mean1 = torch.mean(x_i_tensor,dim=1)
        
    #     # Mean 2
    #     weights_v_c = x_i_tensor.new_tensor([0.25,0.25,0.25,0.25]).repeat(bs,1)
    #     weights_i_c = x_i_tensor.new_tensor([0.25,0.25,0.25,0.25]).repeat(bs,1)
    #     x_v_out_mean2 = (weights_v_c * x_v_tensor.permute(2,3,4,0,1)).permute(3,4,0,1,2).sum(dim=1)
    #     x_i_out_mean2 = (weights_i_c * x_i_tensor.permute(2,3,4,0,1)).permute(3,4,0,1,2).sum(dim=1)

    #     assert torch.equal(x_v_out_mean1,x_v_out_mean2),'x_v_out_mean1 is not equivalent to x_v_out_mean2'
    #     assert torch.equal(x_i_out_mean1,x_i_out_mean2),'x_i_out_mean1 is not equivalent to x_i_out_mean2'

    #     return tuple((x_v_out_mean1, x_i_out_mean1))


    def extract_feat(self, img, img_metas):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        Args:
            img (torch.Tensor): Input images of shape (N, Num_Cam, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            torch.Tensor: of shape (N, C_out, N_x, N_y, N_z)
        """

        batch_size = img.shape[0]
        x_v, x_i = self.extract_img_feat(img, img_metas)

        # x_v = self.dcn_v(x_v)
        # x_i = self.dcn_i(x_i)

        # from IPython import embed
        # embed(header='VICFuser_Voxel_MS.extract_feat')

 
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

        # x_concat = torch.cat((x_v,x_i),dim=1)
        # x = self.conv_o(x_concat)
        

        # from IPython import embed
        # embed(header='xxx')

        x_stack = torch.stack((x_v,x_i),dim=0)
        x = torch.mean(x_stack,dim=0)
        
        # x [bs,C, X, Y, Z] [2,64,288,248,12]
        x = self.neck_3d(x)   
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
