# HW3-Instance-Segmentation

*Student ID: 309553007*

### **Introduction —**

Using Detectron2 to implement the instance segmentation, and the dataset is tiny VOC dataset which contains only 1,349 training images, 100 test images with 20 common object classes.

### **Hardware —**
* Python 3.7

### **Reproducing Submission —**
> 1. Dataset Preparation
> 2. Train
> 3. Inference

#### 1. Dataset Preparation
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

#### 2. Train
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

* Step 4: Set configs for training
```
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("Tiny_dataset", )
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = (
    15000
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
```

* Step 5: Start training
```
from detectron2.engine import DefaultTrainer

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

*If you want to resume training, you could set ```trainer.resume_or_load(resume=True)```*


#### 3. Inference
* Step 1: Set configs for testing
```
cfg.MODEL.WEIGHTS = "../model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55    
```

* Step2: Set model predictor and use ```pycocotools``` to read  ```test.json```
```
from detectron2.engine import DefaultPredictor
from pycocotools.coco import COCO

predictor = DefaultPredictor(cfg)
cocoGt = COCO("../test.json")
```

* Step3: Use model predictor to predict evey testimg images

### **Reference —**
https://github.com/facebookresearch/detectron2
https://github.com/PeiChunChang/NCTU-VRusingDL/tree/master/HW4-instance-segmentation
