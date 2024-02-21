import os
import numpy as np

# from mmdet.datasets import DATASETS
from .builder import DATASETS
from .kitti_dataset import KittiDataset
from .dataset_wrappers import MultiViewMixin
from os import path as osp

from ..core.bbox import LiDARInstance3DBoxes
from ..core import show_multi_modality_result
from mmcv.utils import print_log
from mmdet3d.core.evaluation.v2x_utils.eval_utils import Evaluator
import mmcv
import json
from mmdet3d.datasets.pipelines.transforms_3d import ObjectRangeFilter

t_noise_range = 0.0

@DATASETS.register_module()
class DAIR_VIC_Dataset(KittiDataset):
    def get_data_info(self, index):
        
        # from IPython import embed
        # embed(header='DAIR_VIC_Dataset.get_data_info')
        
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_2_filename = os.path.join(self.data_root, info['image']['image_path'])
        img_3_filename = os.path.join(self.data_root, info['image']['inf_image_path'])
        assert img_2_filename != img_3_filename

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        inf_Trv2c = info['calib']['inf_Tr_velo_to_cam'].astype(np.float32)
        world2veh_lidar = info['calib']['world2veh_lidar'].astype(np.float32)
        world2inf_lidar = info['calib']['world2inf_lidar'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        P3 = info['calib']['P3'].astype(np.float32)
        
        # from IPython import embed
        # embed(header='noise')
        # print(t_noise_range)
        t_noise = (np.array([np.random.normal(0, 1/3), np.random.normal(0, 1/3), 0.0]) * t_noise_range)
        world2inf_lidar = self.trans_noise(world2inf_lidar, t_noise)

        veh_lidar2veh_cam = rect @ Trv2c
        veh_lidar2world = np.linalg.inv(world2veh_lidar)
        veh_lidar2inf_cam = rect @ inf_Trv2c @  world2inf_lidar @ veh_lidar2world
        
        veh_intrinsic = np.copy(P2)
        inf_intrinsic = np.copy(P3)
        # assert np.allclose(intrinsic_2, intrinsic_3)

        input_dict = dict(
            sample_idx=sample_idx,
            img_prefix=[None, None],
            img_info=[dict(filename=img_2_filename), dict(filename=img_3_filename)],
            lidar2img=dict(
                extrinsic=[veh_lidar2veh_cam, veh_lidar2inf_cam],
                intrinsic=[veh_intrinsic,inf_intrinsic]
            )
        )


        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        # if not self.test_mode:
        #     annos = self.get_ann_info(index)
        #     input_dict['ann_info'] = annos
        return input_dict
    
    def trans_noise(self, world2inf_lidar, t_noise):
        assert world2inf_lidar.shape == (4,4)
        assert t_noise.shape == (3,)

        world2inf_lidar[0:3,3] = world2inf_lidar[0:3,3] + t_noise

        return world2inf_lidar


    def eulerAnglesToRotationMatrix(self, theta):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta[0]), -np.sin(theta[0])],
                        [0, np.sin(theta[0]), np.cos(theta[0])]
                        ])

        R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                        [0, 1, 0],
                        [-np.sin(theta[1]), 0, np.cos(theta[1])]
                        ])

        R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                        [np.sin(theta[2]), np.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))
        # print(f"Rotate matrix:\n{R}")
        return R

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """



        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['velodyne_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            img_metas, img = self._extract_data(
                i, pipeline, ['img_metas', 'img'])

            # gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()

            ids = list(['003150','004105','010813','010948','010989','011530','011867','013848','019940','019947','019981'])
            # ids = list(['011530','010813'])
            # ids = list(['011530'])
            if file_name in ids:
                print('file_name:',file_name)
            else:
                continue
            # from IPython import embed
            # embed(header='gt_bboxes')

            assert isinstance(pipeline.transforms[2],ObjectRangeFilter)
            anno = self.get_ann_info(i)
            anno_f = pipeline.transforms[2](anno)
            gt_bboxes = anno_f['gt_bboxes_3d'].tensor.numpy()


            pred = dict()
            pred['gt_bboxes_3d'] = result['boxes_3d']
            pred['gt_labels_3d'] = result['labels_3d']
            pred_f = pipeline.transforms[2](pred)
            pred_bboxes = pred_f['gt_bboxes_3d'].tensor.numpy()
            # pred_bboxes = result['boxes_3d'].tensor.numpy()


            # multi-modality visualization
            if self.modality['use_camera'] and 'lidar2img' in img_metas.keys():
                
                img = img.numpy()
                # need to transpose channel to first dim

                # from IPython import embed
                # embed(header='DAIR_VIC_Dataset.show')
                img_v = img[0]
                img_i = img[1]
                img_v = img_v.transpose(1, 2, 0)
                img_i = img_i.transpose(1, 2, 0)
                show_pred_bboxes = LiDARInstance3DBoxes(
                    pred_bboxes, origin=(0.5, 0.5, 0))
                show_gt_bboxes = LiDARInstance3DBoxes(
                    gt_bboxes, origin=(0.5, 0.5, 0))
                

                proj_mat_ex0 = img_metas['lidar2img']['extrinsic'][0]
                proj_mat_in0 = img_metas['lidar2img']['intrinsic'][0]
                proj_mats_v = proj_mat_in0 @ proj_mat_ex0
                proj_mat_ex1 = img_metas['lidar2img']['extrinsic'][1]
                proj_mat_in1 = img_metas['lidar2img']['intrinsic'][1]
                proj_mats_i = proj_mat_in1 @ proj_mat_ex1
                out_dir_v = osp.join(out_dir, 'veh')
                out_dir_i = osp.join(out_dir, 'inf')
                print(file_name)
                print(out_dir_v)
                print(out_dir_i)
                show_multi_modality_result(
                    img_v,
                    show_gt_bboxes,
                    show_pred_bboxes,
                    proj_mats_v,
                    out_dir_v,
                    file_name,
                    box_mode='lidar',
                    show=show,
                    gt_bbox_color=(0, 255, 0),
                    pred_bbox_color=(0, 0, 255))

                show_multi_modality_result(
                    img_i,
                    show_gt_bboxes,
                    show_pred_bboxes,
                    proj_mats_i,
                    out_dir_i,
                    file_name,
                    box_mode='lidar',
                    show=show,
                    gt_bbox_color=(0, 255, 0),
                    pred_bbox_color=(0, 0, 255))

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
                Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        # result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        # from mmdet3d.core.evaluation import kitti_eval
        # gt_annos = [info['annos'] for info in self.data_infos]

        # if isinstance(result_files, dict):
        #     ap_dict = dict()
        #     for name, result_files_ in result_files.items():
        #         eval_types = ['bbox', 'bev', '3d']
        #         if 'img' in name:
        #             eval_types = ['bbox']
        #         ap_result_str, ap_dict_ = kitti_eval(
        #             gt_annos,
        #             result_files_,
        #             self.CLASSES,
        #             eval_types=eval_types)
        #         for ap_type, ap in ap_dict_.items():
        #             ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

        #         print_log(
        #             f'Results of {name}:\n' + ap_result_str, logger=logger)

        # else:
        #     if metric == 'img_bbox':
        #         ap_result_str, ap_dict = kitti_eval(
        #             gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
        #     else:
        #         ap_result_str, ap_dict = kitti_eval(gt_annos, result_files,
        #                                             self.CLASSES)
        #     print_log('\n' + ap_result_str, logger=logger)

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()
        # if show or out_dir:
        #     self.show(results, out_dir, show=show, pipeline=pipeline)

        # from IPython import embed
        # embed(header='DAIR_V2X evaluate')

        logger.info('Enter DAIR_VIC_Dataset.evaluate')
        ap_dict = dict()
        pred_classes = ["car"]
        evaluator = Evaluator(pred_classes)
        prog_bar = mmcv.ProgressBar(len(results))
        for idx, result in enumerate(results):

            anno = self.prepare_test_data(idx)
            # while True:
            #     anno = self.prepare_train_data(idx)
            #     if anno is None:
            #         idx = self._rand_another(idx)
            #         continue
            #     else:
            #         break

            assert anno is not None, f'should use prepare_test_data'
            gt_bboxes_3d = anno['gt_bboxes_3d'].data
            gt_labels_3d = anno['gt_labels_3d'].data
            # else:
            #     from IPython import embed
            #     embed(header='None')
             
            label = dict()
            num_targets = len(gt_bboxes_3d)

            label_3d = gt_labels_3d.detach().cpu().numpy()
            boxes_3d = gt_bboxes_3d.corners.detach().cpu().numpy()
            label['labels_3d'] = label_3d
            label['scores_3d'] = np.ones([num_targets])
            label['boxes_3d'] = boxes_3d
                        
            pred=dict() 
            pred_8points= result['boxes_3d'].corners.detach().cpu().numpy()
            pred['boxes_3d'] = pred_8points
            pred['scores_3d'] = result['scores_3d'].detach().cpu().numpy()
            pred['labels_3d'] = result['labels_3d'].detach().cpu().numpy()

            evaluator.add_frame(pred, label)
                
            
            for _ in range(1):
                prog_bar.update()
        
        print('\n')
        ap_3d = evaluator.print_ap("3d")
        ap_bev = evaluator.print_ap("bev")
        ap_results = dict()
        ap_results['3d'] = ap_3d['3d']
        ap_results['bev'] = ap_bev['bev']

        ap_results_str = json.dumps(ap_results, indent=2)
        print_log('VIC Eval_Results\n' + ap_results_str, logger=logger)


        for ap_key in ap_results.keys():
            for pred_class in ap_results[ap_key].keys():
                for iou in ap_results[ap_key][pred_class].keys():
                    new_key = pred_class+'_'+ap_key+'_'+str(iou)
                    # print(new_key)
                    ap_dict[new_key] = ap_results[ap_key][pred_class][iou]

        
        return ap_dict