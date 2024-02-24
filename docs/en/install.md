# Step-by-step installation instructions


Please refer to [getting_started.md](getting_started.md) for installation, the same as [**MMDetection3d**](https://github.com/open-mmlab/mmdetection3d) Documentation.




**a. Create a conda virtual environment and activate it.**
```shell
conda create -n emiff python=3.7 -y
conda activate emiff
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9

```


**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.6.2
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.25.2
pip install mmsegmentation==0.29.0
```

**e. Clone EMIFF.**
```
git clone https://github.com/Bosszhe/EMIFF.git
```


**f. Install mmdet3d from source code.**
```shell
pip install -e . 
```

