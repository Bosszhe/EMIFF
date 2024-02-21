import numpy as np

from ..builder import PIPELINES
from mmdet3d.datasets.pipelines.compose import Compose
from mmdet.datasets.pipelines import LoadImageFromFile

from PIL import Image
import mmcv
import numpy as np
import copy


def plot_img(a,str):
    img_mean=np.array([123.675, 116.28, 103.53])
    img_std=np.array([58.395, 57.12, 57.375])
    to_rgb = True

    im = mmcv.imdenormalize(a, img_mean, img_mean, to_rgb)
    im_pil = Image.fromarray(im.astype('uint8'), mode='RGB')
    im_pil.save(str+'_img.png')


@PIPELINES.register_module()
class MultiViewPipeline_old:
    def __init__(self, transforms, n_images):
        self.transforms = Compose(transforms)
        self.n_images = n_images
        print('MultiViewPipeline.__init__')

    def __call__(self, results):
        imgs = []
        extrinsics = []
        
        ids = np.arange(self.n_images)
        # ids = np.arange(len(results['img_info']))
        # replace = True if self.n_images > len(ids) else False
        # ids = np.random.choice(ids, self.n_images, replace=replace)
        
        # from IPython import embed
        # embed(header = 'MultiViewPipeline')
        
        # flip_flag = False

        print('results[gt_bboxes_3d]',results['gt_bboxes_3d'])
        for i in ids.tolist():
            

            
            from IPython import embed
            embed(header = 'MultiViewPipeline_before'+ str(i))
            
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            for key in results.keys():
                if key not in ['img_prefix', 'img_info']:
                    _results[key] = results[key]

            # from IPython import embed
            # embed(header = 'MultiViewPipeline_'+ str(i))
            
            # _results = self.transforms(_results)
       
            if i == 0:
                _results = self.transforms(_results)
                
                # from IPython import embed
                # embed(header=str(i))
                flip_flag = _results['flip']
                flip_direction_flag = _results['flip_direction']
                pcd_horizontal_flip_flag = _results['pcd_horizontal_flip']
                scale_flag = _results['scale']
                
                tmp = copy.deepcopy(_results)
                print('tmp[gt_bboxes_3d]',tmp['gt_bboxes_3d'])
                
                
                # for key in _results.keys():
                #         tmp[key] = _results[key]
                
            else:
                _results['flip'] = flip_flag
                _results['flip_direction'] = flip_direction_flag
                _results['scale'] = scale_flag
                _results['pcd_horizontal_flip'] = pcd_horizontal_flip_flag
                _results = self.transforms(_results)
                # from IPython import embed
                # embed(header=str(i))
            
            # from IPython import embed
            # embed(header = 'MultiViewPipeline_after'+ str(i))
            
            print(i)
            # # print(_results)
            print('_results[flip]:'+str(i),_results['flip'])
            print('_results[gt_bboxes_3d]',_results['gt_bboxes_3d'])
            
            # left_img = _results['img']
            # plot_img(left_img,str(i))
            # print('_results[flip_direction]:'+str(i),_results['flip_direction'])
            # print('_results[scale]:'+str(i),_results['scale'])
            # print('_results[pcd_horizontal_flip]:'+str(i),_results['pcd_horizontal_flip'])
            # print('_results[pcd_horizontal_flip]:'+str(i),_results['pcd_horizontal_flip'])
            # print('_results[img].shape:'+str(i),_results['img'].shape)
            # print('_results[pad_shape]:'+str(i),_results['pad_shape'])

                        
                        
            imgs.append(_results['img'])
            extrinsics.append(results['lidar2img']['extrinsic'][i])

        results['img'] = imgs
        # results['lidar2img']['extrinsic'] = extrinsics
        
        print('tmp[gt_bboxes_3d]',tmp['gt_bboxes_3d'])  
        
        for key in tmp.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = tmp[key]
        print('results[gt_bboxes_3d]',results['gt_bboxes_3d'])
        
        from IPython import embed
        embed()
        return results



@PIPELINES.register_module()
class MultiViewPipeline_Test_old:
    def __init__(self, transforms, n_images):
        self.transforms = Compose(transforms)
        self.n_images = n_images
        print('MultiViewPipeline_Test.__init__')

    def __call__(self, results):
        imgs = []
        extrinsics = []
        
        ids = np.arange(self.n_images)
        for i in ids.tolist():
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            for key in results.keys():
                if key not in ['img_prefix', 'img_info']:
                    _results[key] = results[key]
       
            if i == 0:
                _results = self.transforms(_results)
                scale_flag = _results['scale']
            else:
                _results['scale'] = scale_flag
                _results = self.transforms(_results) 
                 
            imgs.append(_results['img'])
            extrinsics.append(results['lidar2img']['extrinsic'][i])
        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs
        # results['lidar2img']['extrinsic'] = extrinsics
        return results
    
    

@PIPELINES.register_module()
class RandomShiftOrigin:
    def __init__(self, std):
        self.std = std

    def __call__(self, results):
        shift = np.random.normal(.0, self.std, 3)
        results['lidar2img']['origin'] += shift
        return results


@PIPELINES.register_module()
class KittiSetOrigin:
    def __init__(self, point_cloud_range):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.origin = (point_cloud_range[:3] + point_cloud_range[3:]) / 2.

    def __call__(self, results):
        results['lidar2img']['origin'] = self.origin.copy()
        return results


@PIPELINES.register_module()
class KittiRandomFlip:
    def __call__(self, results):
        if results['flip']:
            results['lidar2img']['intrinsic'][0, 2] = -results['lidar2img']['intrinsic'][0, 2] + \
                                                      results['ori_shape'][1]
            flip_matrix_0 = np.eye(4, dtype=np.float32)
            flip_matrix_0[0, 0] *= -1
            flip_matrix_1 = np.eye(4, dtype=np.float32)
            flip_matrix_1[1, 1] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0]
            extrinsic = flip_matrix_0 @ extrinsic @ flip_matrix_1.T
            results['lidar2img']['extrinsic'][0] = extrinsic
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 0], -center[:, 1]) - alpha
            center_flip = center
            center_flip[:, 1] *= -1
            alpha_flip = np.arctan2(center_flip[:, 0], -center_flip[:, 1]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results


@PIPELINES.register_module()
class SunRgbdSetOrigin:
    def __call__(self, results):
        intrinsic = results['lidar2img']['intrinsic'][:3, :3]
        extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
        projection = intrinsic @ extrinsic
        h, w, _ = results['ori_shape']
        center_2d_3 = np.array([w / 2, h / 2, 1], dtype=np.float32)
        center_2d_3 *= 3
        origin = np.linalg.inv(projection) @ center_2d_3
        results['lidar2img']['origin'] = origin
        return results


@PIPELINES.register_module()
class SunRgbdTotalLoadImageFromFile(LoadImageFromFile):
    def __call__(self, results):
        file_name = results['img_info']['filename']
        flip = file_name.endswith('_flip.jpg')
        if flip:
            results['img_info']['filename'] = file_name.replace('_flip.jpg', '.jpg')
        results = super().__call__(results)
        if flip:
            results['img'] = results['img'][:, ::-1]
        return results


@PIPELINES.register_module()
class SunRgbdRandomFlip:
    def __call__(self, results):
        if results['flip']:
            flip_matrix = np.eye(3)
            flip_matrix[0, 0] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
            results['lidar2img']['extrinsic'][0][:3, :3] = flip_matrix @ extrinsic @ flip_matrix.T
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 1], center[:, 0]) - alpha
            center_flip = center @ flip_matrix
            alpha_flip = np.arctan2(center_flip[:, 1], center_flip[:, 0]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results
