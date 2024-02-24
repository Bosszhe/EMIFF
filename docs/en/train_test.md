
## Prerequisites

Please ensure you have prepared the environment and the DAIR-V2X-C dataset with KITTI format.


### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]

# eg: python tools/train.py cfgs/vimi_960x540_12e_bs2.py
```

If you want to specify the working directory in the command, you can add an argument `--work-dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

### Evaluation or Test
You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test_vic.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show] [--show-dir ${SHOW_DIR}]
```

### Visualization 

see [visualize_results_dair.py](../../tools/misc/visualize_results_dair.py)