# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from mmcv.image import tensor2imgs
from mmdet3d.models.model_utils import Channel


from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)

from mmdet3d.core.evaluation.v2x_utils.filter_utils import RectFilter
from mmdet3d.core.evaluation.v2x_utils.geometry_utils import range2box
from mmdet3d.core.evaluation.v2x_utils.transformation_utils import get_arrow_end



def get_box_info(result):
    if len(result[0]["boxes_3d"].tensor) == 0:
        box_lidar = np.zeros((1, 8, 3))
        box_ry = np.zeros(1)
    else:
        box_lidar = result[0]["boxes_3d"].corners.numpy()
        box_ry = result[0]["boxes_3d"].tensor[:, -1].numpy()
    box_centers_lidar = box_lidar.mean(axis=1)
    arrow_ends_lidar = get_arrow_end(box_centers_lidar, box_ry)
    return box_lidar, box_ry, box_centers_lidar, arrow_ends_lidar


def single_gpu_test_dair(model,
                    data_loader,
                    evaluator,
                    extended_range=[30, -39.68, -3, 50, 39.68, 1],
                    show=False,
                    out_dir=None,
                    skip=False):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset

    pipe = Channel()
    # offsets=list()

    prog_bar = mmcv.ProgressBar(len(dataset))
    for idx, data in enumerate(data_loader):
        # print('\n')
        # print(idx)
        # print(data['img_metas'].data[0][0]['sample_idx'])
        # if data['img_metas'].data[0][0]['sample_idx'] == 874:
        #     from IPython import embed
        #     embed(header='single_gpu_test_dair')
        
        # if idx == 545:
        #     from IPython import embed
        #     embed(header='545')
    
        if skip:
            if idx % 20 != 0:
                continue
            


        img = data['img'].data[0]
        img_metas = data['img_metas'].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]


        # veh_id = img_metas[0]['sample_idx']

        # veh_id_s = "%06d" % veh_id
        # # ids = list(['003150','004105','010813','010948','010989','011530','011867','013848','019940','019947','019981'])
        # ids = list(['010813'])
        # if veh_id_s in ids:
        #     print('veh_id:',veh_id_s)
        # else:
        #     continue

        batch_size = img.shape[0]

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, pipe=pipe, **data)
            
        # from IPython import embed
        # embed(header='single_gpu_test_dair')
        extended_range = [0, -39.68, -3, 100, 39.68, 1]
        box, box_ry, box_center, arrow_ends = get_box_info(result)
        extended_range_8points = range2box(np.array(extended_range))
        pred_filter = RectFilter(extended_range_8points[0])

        for i in range(batch_size):
            
            label = dict()

            num_targets = len(gt_bboxes_3d[i])
            veh_id = img_metas[i]['sample_idx']            
            label_3d = gt_labels_3d[i].detach().cpu().numpy()
            boxes_3d = gt_bboxes_3d[i].corners.detach().cpu().numpy()

            label['labels_3d'] = label_3d
            label['scores_3d'] = np.ones([num_targets])
            # label['boxes_3d'] = boxes_3d[:,[3,7,4,0,2,6,5,1],:]
            label['boxes_3d'] = boxes_3d
                      
            pred=dict() 
            # if len(result[i]['boxes_3d'].tensor) == 0:
            #     # from IPython import embed
            #     # embed(header='results == 0')
            #     print('The results of batch id.%d is None' %(i))
            #     pred['boxes_3d'] = result[i]['boxes_3d'].tensor.detach().cpu().numpy()
            # else:
            # pred_8points= result[i]['boxes_3d'].corners.detach().cpu().numpy()
            # pred['boxes_3d'] = pred_8points

            
            # pred['scores_3d'] = result[i]['scores_3d'].detach().cpu().numpy()
            # pred['labels_3d'] = result[i]['labels_3d'].detach().cpu().numpy()

            # from IPython import embed
            # embed(header='xxxxxxxx')
            # Filter out labels
            remain = []
            if len(result[i]["boxes_3d"].tensor) != 0:
                for index in range(box.shape[0]):
                    if pred_filter(box[index]):
                        remain.append(index)

            if len(remain) >= 1:
                box = box[remain]
                box_center = box_center[remain]
                arrow_ends = arrow_ends[remain]
                pred["scores_3d"] = result[i]["scores_3d"].numpy()[remain]
                pred["labels_3d"] = result[i]["labels_3d"].numpy()[remain]
            else:

                # from IPython import embed
                # embed(header='bbbbb')
                box = np.zeros((1, 8, 3))
                box_center = np.zeros((1, 1, 3))
                arrow_ends = np.zeros((1, 1, 3))
                pred["labels_3d"] = np.zeros((1))
                pred["scores_3d"] = np.zeros((1))

            pred['boxes_3d'] = box


            # # Count CenterPoint Distribution
            # gt_boxes = gt_bboxes_3d[i].tensor.detach().cpu().numpy()
            # pred_boxes = result[i]['boxes_3d'].tensor.detach().cpu().numpy()
            # for i, pred_box in enumerate(pred_boxes):
            #     delta_c = gt_boxes[:,:2] - pred_box[:2]
            #     offsets.append(np.linalg.norm(delta_c, ord=2, axis=1).min()) 
            proj_mat_ex0 = img_metas[i]['lidar2img']['extrinsic'][0]
            proj_mat_in0 = img_metas[i]['lidar2img']['intrinsic'][0]
            veh_lidar2veh_cam = proj_mat_in0 @ proj_mat_ex0
            proj_mat_ex1 = img_metas[i]['lidar2img']['extrinsic'][1]
            proj_mat_in1 = img_metas[i]['lidar2img']['intrinsic'][1]
            veh_lidar2inf_cam = proj_mat_in1 @ proj_mat_ex1

            inf_cam2veh_lidar = np.linalg.inv(veh_lidar2inf_cam)
            s_point = np.array([0,0,0,1]).reshape(4,1)
            inf_c_veh_lidar = inf_cam2veh_lidar @ s_point

            if show:
                pp = pred['boxes_3d']
                ll = label['boxes_3d']
                veh_id_str = "%06d" % veh_id
                vis_pred_label_2(pp,ll,veh_id_str+'_'+str(i),s_point,inf_c_veh_lidar,out_dir) 

            result[i]['label_boxes_3d'] = label['boxes_3d']
            result[i]['pred_boxes_3d'] = pred['boxes_3d']
            evaluator.add_frame(pred, label)
            pipe.flush()
            
        
        results.extend(result)

        bs = len(result)
        if skip:
            bs = bs*20
        for _ in range(bs):
            prog_bar.update()
    
    # from IPython import embed
    # embed(header='off')

    print('\n')
    ap_3d = evaluator.print_ap("3d")
    ap_bev = evaluator.print_ap("bev")
    ap_results = dict()
    ap_results['3d'] = ap_3d['3d']
    ap_results['bev'] = ap_bev['bev']
    print("Average Communication Cost = %.2lf Bytes" % (pipe.average_bytes()))
        

        
    return results, ap_results


# def vis_pred_label(pred,label,str,out_dir):
        
#         img_save_floder_path = osp.join(out_dir,'vis_results_dair','pred_label_imgs')
#         if not osp.exists(img_save_floder_path):
#             os.makedirs(img_save_floder_path)
            
#         plt.cla()
#         for i in range(label.shape[0]):
#             x3 = label[i,[0,4,7,3,0],0]  
#             y3 = label[i,[0,4,7,3,0],1]  
#             plt.plot(x3,y3,'g')
#             # plt.scatter(label[i,3,0],label[i,3,1],c='g',marker='o')
#             plt.axis('equal')   
            
#         for i in range(pred.shape[0]):
#             x = pred[i,[0,4,7,3,0],0]  
#             y = pred[i,[0,4,7,3,0],1]  
#             plt.plot(x,y,'r')
#             # plt.scatter(pred[i,0,0],pred[i,0,1],c='r',marker='*')
#             plt.axis('equal')   
        
#         img_save_path = osp.join(img_save_floder_path,str+'pred_label.png')
#         plt.savefig(img_save_path)
    
def vis_pred_label_2(pred, label, str, veh_c_s_world, inf_c_s_world, out_dir):
        
        img_save_floder_path = osp.join(out_dir,'vis_results_dair','BEV_pred_label')
        if not osp.exists(img_save_floder_path):
            os.makedirs(img_save_floder_path)
            
        plt.cla()
        plt.scatter(-veh_c_s_world[1,0],veh_c_s_world[0,0],c='b',marker='o')
        # plt.scatter(-inf_c_s_world[1,0],inf_c_s_world[0,0],c='c',marker='^')
        for i in range(label.shape[0]):
            x3 = label[i,[0,4,7,3,0],0]  
            y3 = label[i,[0,4,7,3,0],1]  
            plt.plot(-y3,x3,'g')
            # plt.scatter(label[i,3,0],label[i,3,1],c='g',marker='o')
            # plt.axis('equal')   
            
        for i in range(pred.shape[0]):
            x = pred[i,[0,4,7,3,0],0]  
            y = pred[i,[0,4,7,3,0],1]  
            plt.plot(-y,x,'r')
            # plt.scatter(pred[i,0,0],pred[i,0,1],c='r',marker='*')
        
        plt.axis('equal')

        plt.ylim([-5, 65])
        plt.xlim([-40, 30])

        img_save_path = osp.join(img_save_floder_path,str+'pred_label.png')
        plt.savefig(img_save_path,dpi=300)
        
# def vis_fun(pred,label,str,out_dir):
        
#         img_save_floder_path = osp.join(out_dir,'vis_results_dair','pred_label_imgs')
#         if not osp.exists(img_save_floder_path):
#             os.makedirs(img_save_floder_path)
            
#         plt.cla()
#         for i in range(label.shape[0]):
#             x3 = label[i,[3,2,1,0],0]  
#             y3 = label[i,[3,2,1,0],1]  
#             plt.plot(x3,y3,'g')
#             plt.scatter(label[i,3,0],label[i,3,1],c='g',marker='o')
#             plt.axis('equal')   
            
#         for i in range(pred.shape[0]):
#             x = pred[i,[0,4,7,3],0]  
#             y = pred[i,[0,4,7,3],1]  
#             plt.plot(x,y,'r')
#             plt.scatter(pred[i,0,0],pred[i,0,1],c='r',marker='*')
#             plt.axis('equal')   
        
#         img_save_path = osp.join(img_save_floder_path,str+'pred_label.png')
#         plt.savefig(img_save_path)