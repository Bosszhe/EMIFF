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
import math
from mmdet3d.models.model_utils.naive_compressor import NaiveCompressor_UNet
from mmdet3d.models.detectors.vicfuser_voxel.vicfuser_voxel_ccm import Mlp, CCMNet


def attention(query, key, mask=None, dropout=None):

    # from IPython import embed
    # embed()

    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn

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

class MultiScaleBlock(nn.Module):
    
    def __init__(self, in_channels,out_channels):
        super(MultiScaleBlock, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
                
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        return tuple([x0,x1,x2,x3])


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

class SS_NaiveCompressor(nn.Module):

    def __init__(self, input_dim, c_compress_ratio, s_compress_ratio):
        super(SS_NaiveCompressor, self).__init__()

        # from IPython import embed
        # embed(header='MSC')

        assert (s_compress_ratio<=4) & (s_compress_ratio >=0)
        s_compress_ratio0 = max(s_compress_ratio,0)
        self.compressor0 = NaiveCompressor_UNet(input_dim,c_compress_ratio,s_compress_ratio0)

    def forward(self, x):
        x0 = self.compressor0(x)

        return x0




@DETECTORS.register_module()
class VICFuser_Voxel_MSCA_C_CCM(BaseDetector):
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
                 se_reduction_ratio=1,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 neck_dcn=None):
        print('VICFuser_Voxel_MSCA_C_CCM_0129.__init__')
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
        self.c_compress_ratio = compress_ratio
        self.s_compress_ratio = s_compress_ratio
        self.img_feat_channels = neck.out_channels
        self.img_feat_channels_c = self.img_feat_channels // self.c_compress_ratio
        self.se_reduction_ratio = se_reduction_ratio

        self.dcn_up_conv_v = DCN_Up_Conv_List(neck_dcn, self.img_feat_channels)
        self.dcn_up_conv_i = DCN_Up_Conv_List(neck_dcn, self.img_feat_channels)

        self.inf_compressor = SS_NaiveCompressor(self.img_feat_channels, self.c_compress_ratio, self.s_compress_ratio)
        self.ms_block_inf = MultiScaleBlock(self.img_feat_channels,self.img_feat_channels)

        self.ccmnet =  CCMNet(self.img_feat_channels,self.img_feat_channels,self.img_feat_channels,self.se_reduction_ratio)



    def extract_img_feat(self, img, img_metas):
        """Extract features from images."""
        bs = img.shape[0]
        img_v = img[:,0,...]
        img_i = img[:,1,...]
        
        
        x_v = self.backbone_v(img_v)
        x_v = self.neck_v(x_v)
        x_v = self.dcn_up_conv_v(list(x_v))
        x_v_tensor = torch.stack(x_v).permute(1,0,2,3,4)
        x_v_out = torch.mean(x_v_tensor,dim=1)

        x_i = self.backbone_i(img_i)
        x_i0 = self.neck_i(x_i)[0]
        # from IPython import embed
        # embed(header='compress')

        # Add compression encoder-decoder
        x_i0 = self.inf_compressor(x_i0)

        # from IPython import embed
        # embed(header='after comp')

        x_i = self.ms_block_inf(x_i0)

        x_i = self.dcn_up_conv_i(list(x_i))
        x_i_tensor = torch.stack(x_i).permute(1,0,2,3,4)

        # query.shape[B,C]
        # key.shape[B,N_levels,C]
        query = torch.mean(x_v_out,dim=(-2,-1))[:,None,:]
        key = torch.mean(x_i_tensor,dim=(-2,-1))
        weights_i = attention(query,key).squeeze(1)

        # print('attention_weights',weights_i)
        
        x_i_out = (weights_i[:,:,None,None,None] * x_i_tensor).sum(dim=1)

        return tuple((x_v_out, x_i_out))

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

        x_v, x_i = self.ccmnet(x_v, x_i, img_metas)

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
        
        # x [bs,C, X, Y, Z] [2,64,248,288,12]
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
