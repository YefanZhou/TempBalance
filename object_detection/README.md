# Experiments on Object Detection

## Install
```bash
    conda create -n tb_yolo python=3.8 
    conda activate tb_yolo 
    pip install -r requirements.txt 
```

## Dataset setup
Find the 'TESTVOC2007.yaml', 'VALVOC2007.yaml' and 'VOC2007.yaml' three files in the directory [YOLOv8/ultralytics/datasets](https://github.com/YefanZhou/TempBalance/object_detection/src/YOLOv8/ultralytics/datasets), and put the path (absolute path) in <PUT_YOUR_PATH_HERE>. If datasets are not exist, they will be downloaded and unzipped automatically.

## Run Training scripts
```bash
    cd src
    # baseline
    bash  scripts/voc2007_train_baseline.sh
    # ours
    bash  scripts/voc2007_train_tb.sh
```

## Run test scripts for selecting learning rate
```bash
    cd src
    # baseline
    bash scripts/voc2007_test_baseline.sh
    # ours
    bash  scripts/voc2007_test_tb.sh
```


## Acknowledgement
1. [Yuanzhe Hu](https://github.com/HUST-AI-HYZ) is the main contributor to the object detection implementation.
2. We thank the open-sourced package [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
2. For hyperparameters not specified in the provided bash scripts, please refer to the defaults set in  [runtrain.py](https://github.com/YefanZhou/TempBalance/object_detection/src/YOLOv8/runtrain.py). 
3. The original hyperparameter 'Optimizer' in Ultralytics Code is replaced by 'optim_type'.


