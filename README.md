# EMIFF: Enhanced Multi-scale Image Feature Fusion for Vehicle-Infrastructure Cooperative 3D Object Detection

## Abstract

In autonomous driving, cooperative perception makes use of multi-view cameras from both vehicles and infrastructure, providing a global vantage point with rich semantic context of road conditions beyond a single vehicle viewpoint. Currently, two major challenges persist in vehicle-infrastructure cooperative 3D (VIC3D) object detection: $1)$ inherent pose errors when fusing multi-view images, caused by time asynchrony across cameras;  $2)$ information loss in transmission process resulted from limited communication bandwidth.
To address these issues, we propose a novel camera-based 3D detection framework for VIC3D task, \textit{Enhanced Multi-scale Image Feature Fusion} (EMIFF).
To fully exploit holistic perspectives from both vehicles and infrastructure, we propose \textit{Multi-scale Cross Attention} (MCA) and \textit{Camera-aware Channel Masking} (CCM) modules to enhance infrastructure and vehicle features at scale, spatial, and channel levels to correct the pose error introduced by camera asynchrony. We also introduce a \textit{Feature Compression} (FC) module with channel and spatial compression blocks for transmission efficiency. Experiments show that EMIFF achieves SOTA on DAIR-V2X-C datasets, significantly outperforming previous early-fusion and late-fusion methods with comparable transmission costs.


## Code

**Code and pretrained models will be released upon publication.**

## VIMI_Architecture
![Architecture](./Fig/VIMI_architecture.png)


## VIMI_Performance
* DAIR-V2X-C

![performance](./Fig/VIMI_performance.png)

* Compression Impact

<!-- ![compression](./Fig/CM_3D.png =960x540) -->

<img src="./Fig/CM_3D.png" width="480" height="270">



## Citation

If you find our work useful in your research, please consider citing:

```
@misc{wang2023vimi,
      title={VIMI: Vehicle-Infrastructure Multi-view Intermediate Fusion for Camera-based 3D Object Detection}, 
      author={Zhe Wang and Siqi Fan and Xiaoliang Huo and Tongda Xu and Yan Wang and Jingjing Liu and Yilun Chen and Ya-Qin Zhang},
      year={2023},
      eprint={2303.10975},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
