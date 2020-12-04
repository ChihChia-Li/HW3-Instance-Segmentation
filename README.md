# HW3-Instance-Segmentation

*Student ID: 309553007*

**Introduction —**
Using Detectron2 to implement the instance segmentation, and the dataset is tiny VOC dataset which contains only 1,349 training images, 100 test images with 20 common object classes.

**Hardware —**
* Python 3.7

**Reproducing Submission —**
> 1. Dataset Preparation
> 2. Train
> 3. Inference

1. Dataset Preparation
* Data structure:

```
DL_HW3/
  train_images/
    -2007_000033.jpg
    -2007_000042.jpg
    ...
  test_images/
    -2007_000629.jpg
    -2007_001175.jpg
    ...
   pascal_train.json
   test.json
```

2. Train
* Step 1: 
Setup environment and detectron2
```
pip install -U torch torchvision
pip install git+https://github.com/facebookresearch/fvcore.git
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```

* Step 2: Restart the runtime in order to use newly installed versions

* Step 3: Register the tiny VOC dataset to detectron2
```
from detectron2.data.datasets import register_coco_instances

register_coco_instances("Tiny_dataset", {}, "../pascal_train.json", "../train_images")
```
