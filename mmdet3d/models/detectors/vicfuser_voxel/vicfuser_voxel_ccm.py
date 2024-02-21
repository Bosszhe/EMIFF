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
import numpy as np


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


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super(SELayer,self).__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):

        # from IPython import embed 
        # embed(header='SELayer')
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class SE_Inception_Layer(nn.Module):
    def __init__(self, channels, reduction_ratio=1, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super(SE_Inception_Layer,self).__init__()
        self.conv_reduce = nn.Linear(channels, channels//reduction_ratio)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels//reduction_ratio, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):

        # from IPython import embed 
        # embed(header='SE_Inception_Layer')
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)[...,None,None]


class CASELayer(nn.Module):
    def __init__(self, channels, reduction_ratio=1, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super(CASELayer,self).__init__()
        self.FC_reduce = nn.Linear(channels, channels//reduction_ratio)
        self.act1 = act_layer()
        self.FC_expand = nn.Linear(channels//reduction_ratio, channels)
        self.FC_out = nn.Linear(channels*2, channels)
        self.gate = gate_layer()
        self.max_pooling = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, x, x_se):

        # from IPython import embed 
        # embed(header='CASELayer')

        # x_channel = torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        x_channel = self.max_pooling(x)
        x_se = self.FC_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.FC_expand(x_se)
        x_se = torch.cat((x_channel[...,0,0], x_se), dim=1)
        x_se = self.FC_out(x_se)

        return x * self.gate(x_se)[...,None,None]

class CCMNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, reduction_ratio=1):
        super(CCMNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(16)
        self.context_mlp = Mlp(16, mid_channels, mid_channels)
        self.context_se = SE_Inception_Layer(mid_channels,reduction_ratio=reduction_ratio)  # NOTE: add camera-aware

        # self.context_se = CASELayer(mid_channels,reduction_ratio=8)  # NOTE: add camera-aware
    
    def ida_mat_cal(self,img_meta):
        img_scale_factor = (img_meta['scale_factor'][:2]
                if 'scale_factor' in img_meta.keys() else 1)

        img_shape = img_meta['img_shape'][:2]
        orig_h, orig_w = img_shape

        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        ida_rot *= img_scale_factor
        # ida_tran -= torch.Tensor(crop[:2])
        if 'flip' in img_meta.keys() and img_meta['flip']:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([orig_w, 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b

        ida_mat = ida_rot.new_zeros(4, 4)
        ida_mat[3, 3] = 1
        ida_mat[2, 2] = 1
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 3] = ida_tran

        return ida_mat

    def forward(self, x_v, x_i, img_metas):
        # x [bs,num_cams,C,H,W]
        bs, C, H, W = x_v.shape
        num_cams = 2

        x = torch.stack((x_v,x_i),dim=1).reshape(-1, C, H, W)


        extrinsic_v_list = list()
        extrinsic_i_list = list()
        intrinsic_v_list = list()
        intrinsic_i_list = list()
        for img_meta in img_metas:

            extrinsic_v = torch.Tensor(img_meta['lidar2img']['extrinsic'][0])
            extrinsic_i = torch.Tensor(img_meta['lidar2img']['extrinsic'][1])
            intrinsic_v = torch.Tensor(img_meta['lidar2img']['intrinsic'][0])
            intrinsic_i = torch.Tensor(img_meta['lidar2img']['intrinsic'][1])
            # from IPython import embed
            # embed(header='ida')
            ida_mat = self.ida_mat_cal(img_meta)

            intrinsic_v = ida_mat @ intrinsic_v
            intrinsic_i = ida_mat @ intrinsic_i

            extrinsic_v_list.append(extrinsic_v)
            extrinsic_i_list.append(extrinsic_i)
            intrinsic_v_list.append(intrinsic_v)
            intrinsic_i_list.append(intrinsic_i)

            

        extrinsic_v = torch.stack(extrinsic_v_list)
        extrinsic_i = torch.stack(extrinsic_i_list)
        intrinsic_v = torch.stack(intrinsic_v_list)
        intrinsic_i = torch.stack(intrinsic_i_list)

        extrinsic = torch.stack((extrinsic_v,extrinsic_i),dim=1) 
        intrinsic = torch.stack((intrinsic_v,intrinsic_i),dim=1) 

        in_mlp = torch.stack(
                    (
                        intrinsic[..., 0, 0],
                        intrinsic[..., 1, 1],
                        intrinsic[..., 0, 2],
                        intrinsic[ ..., 1, 2],
                    ),
                    dim=-1
                )

        # from IPython import embed
        # embed(header='DCMNet')
        ex_mlp = extrinsic[...,:3,:].view(bs,num_cams,-1)
        mlp_input = torch.cat((in_mlp,ex_mlp),dim=-1)
        mlp_input = mlp_input.reshape(-1,mlp_input.shape[-1]).to(x.device)

        mlp_input = self.bn(mlp_input)
        x = self.reduce_conv(x)
        # context_se = self.context_mlp(mlp_input)[..., None, None]
        context_se = self.context_mlp(mlp_input)
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        context = context.reshape(bs,num_cams,C,H,W)
        x_v_out = context[:,0,...]
        x_i_out = context[:,1,...]

        # from IPython import embed
        # embed(header='DCMNet end')
        return tuple((x_v_out, x_i_out))



@DETECTORS.register_module()
class VICFuser_Voxel_CCM(BaseDetector):
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
        print('VICFuser_Voxel_CCM_0120.__init__')
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

        self.img_feat_channels = neck.out_channels

        self.dcn_up_conv_v = DCN_Up_Conv_List(neck_dcn, neck.out_channels)
        self.dcn_up_conv_i = DCN_Up_Conv_List(neck_dcn, neck.out_channels)

        # self.confnet_v = double_conv(self.img_feat_channels+1, 1)
        # self.confnet_i = double_conv(self.img_feat_channels+1, 1)
        self.ccmnet =  CCMNet(self.img_feat_channels,self.img_feat_channels*4,self.img_feat_channels)


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

    #     # Mean 1
    #     x_v_out_mean1 = torch.mean(x_v_tensor,dim=1)
    #     x_i_out_mean1 = torch.mean(x_i_tensor,dim=1)
        
    #     return tuple((x_v_out_mean1, x_i_out_mean1))

    def extract_img_feat(self, img, img_metas):
        """Extract features from images."""
        bs = img.shape[0]
        img_v = img[:,0,...]
        img_i = img[:,1,...]
        
        # from IPython import embed
        # embed(header='test')
        x_v = self.backbone_v(img_v)
        x_v = self.neck_v(x_v)
        x_v = self.dcn_up_conv_v(list(x_v))
        x_v_tensor = torch.stack(x_v).permute(1,0,2,3,4)
        x_v_out = torch.mean(x_v_tensor,dim=1)

        x_i = self.backbone_i(img_i)
        x_i = self.neck_i(x_i)
        x_i = self.dcn_up_conv_i(list(x_i))
        x_i_tensor = torch.stack(x_i).permute(1,0,2,3,4)

        # query.shape[B,C]
        # key.shape[B,N_levels,C]
        query = torch.mean(x_v_out,dim=(-2,-1))[:,None,:]
        key = torch.mean(x_i_tensor,dim=(-2,-1))
        weights_i = attention(query,key).squeeze(1)
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
        # from IPython import embed
        # embed(header='distance')
        
        batch_size = img.shape[0]
        x_v, x_i = self.extract_img_feat(img, img_metas)

        x_v, x_i = self.ccmnet(x_v, x_i, img_metas)

        points = self.anchor_generator.grid_anchors(
            [self.n_voxels[::-1]], device=img.device)[0][:, :3]

        # DCM
        # dis_veh = list()
        # dis_inf = list()
        # for i in range(batch_size):
        #     source = np.array([0,0,0,1])
        #     veh_cam2veh_lidar = np.linalg.inv(img_metas[i]['lidar2img']['extrinsic'][0])
        #     veh_cam =  (veh_cam2veh_lidar @ source)[:3]

        #     inf_cam2veh_lidar = np.linalg.inv(img_metas[i]['lidar2img']['extrinsic'][1])
        #     inf_cam =  (inf_cam2veh_lidar @ source)[:3]
        #     veh_cam = points.new_tensor(veh_cam)
        #     inf_cam = points.new_tensor(inf_cam)

        #     d_veh = torch.norm((points - veh_cam),p=2,dim=1)
        #     d_inf = torch.norm((points - inf_cam),p=2,dim=1)

        #     dis_veh.append(d_veh.reshape(self.n_voxels[::-1]))
        #     dis_inf.append(d_inf.reshape(self.n_voxels[::-1]))
        
        # distance_veh = torch.stack(dis_veh)
        # distance_inf = torch.stack(dis_inf)

        # # from IPython import embed
        # # embed(header='dis')

        # distance_veh_bev = torch.mean(distance_veh,dim=1)
        # distance_inf_bev = torch.mean(distance_inf,dim=1)

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

        # x_v_bev = torch.mean(x_v,dim=4)
        # x_i_bev = torch.mean(x_i,dim=4)

        # distance_veh_bev = distance_veh_bev.permute(0,2,1).unsqueeze(1)
        # distance_inf_bev = distance_inf_bev.permute(0,2,1).unsqueeze(1)

        # x_veh_d = torch.cat((x_v_bev,distance_veh_bev),dim=1)
        # x_inf_d = torch.cat((x_i_bev,distance_inf_bev),dim=1)

        # confidences_veh = self.confnet_v(x_veh_d)
        # confidences_inf = self.confnet_i(x_inf_d)

        # x_v = x_v * confidences_veh.unsqueeze(-1)
        # x_i = x_i * confidences_inf.unsqueeze(-1)

        x_stack = torch.stack((x_v,x_i),dim=0)
        x = torch.mean(x_stack,dim=0)
        
        # x [bs,C, X, Y, Z] [2,64,248,288,12]
        x = self.neck_3d(x)   
        # x[0] [bs,C, Y, X] [2,256,288,248]


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
        # embed(header='VICFuser_Voxel_Distance.forward_train')
        
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
